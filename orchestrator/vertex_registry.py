"""
VertexRegistry — manages the multi-agent system on Vertex AI Agent Engine.

The full system (Triage + all specialists) is deployed as a SINGLE
Vertex AI Reasoning Engine resource. This follows the standard ADK pattern:
  - root_agent (with sub_agents) → AdkApp → agent_engines.create()

Advantages over per-agent deployment:
  - Inter-agent routing works (transfer_to_agent operates within the tree)
  - Single resource visible in Vertex AI console with the full system
  - Cloud Trace shows per-agent traces within each conversation
  - More robust: agents are not isolated entities without context
  - Follows the standard ADK convention

Resource name: "stayforlong-multiagent"

CLI:
    python -m orchestrator.provision             # deploy if not exists
    python -m orchestrator.provision --force     # redeploy (update instructions)
    python -m orchestrator.provision --list      # show current state
    python -m orchestrator.provision --delete    # delete resource
    python -m orchestrator.provision --purge-orphans  # clean up previous resources
"""
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

_SYSTEM_DISPLAY_NAME = "stayforlong-multiagent"
_ORPHAN_PREFIX = "stayforlong-agent-"  # prefix of resources from a previous attempt


@dataclass
class SystemResource:
    resource_name: str  # full path: projects/P/locations/L/reasoningEngines/N
    numeric_id: str     # numeric portion only: "1234567890123456"
    display_name: str


class VertexRegistry:
    """
    Manages the Vertex AI Reasoning Engine resource for the multi-agent system.

    The full tree (root_agent with all sub_agents) is deployed as a single
    resource named "stayforlong-multiagent".

    Local code (agents/, mock_data/, config.py) is bundled as extra_packages
    alongside a generated _vertex_env.py that injects config values at runtime.
    """

    def __init__(self):
        self._initialized = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_system(self) -> Optional[SystemResource]:
        """Return the system resource if deployed, or None."""
        self._init()
        from vertexai import agent_engines
        try:
            for engine in agent_engines.list():
                dn = getattr(engine._gca_resource, "display_name", "") or ""
                if dn == _SYSTEM_DISPLAY_NAME:
                    rn = engine.resource_name
                    return SystemResource(
                        resource_name=rn,
                        numeric_id=rn.split("/")[-1],
                        display_name=dn,
                    )
        except Exception:
            logger.exception("Error looking up system in Vertex AI")
        return None

    def list_all(self) -> list[SystemResource]:
        """List ALL stayforlong-* resources (system + possible orphans)."""
        self._init()
        from vertexai import agent_engines
        results = []
        try:
            for engine in agent_engines.list():
                dn = getattr(engine._gca_resource, "display_name", "") or ""
                if dn == _SYSTEM_DISPLAY_NAME or dn.startswith(_ORPHAN_PREFIX):
                    rn = engine.resource_name
                    results.append(SystemResource(
                        resource_name=rn,
                        numeric_id=rn.split("/")[-1],
                        display_name=dn,
                    ))
        except Exception:
            logger.exception("Error listing Vertex AI resources")
        return results

    def deploy_system(self, root_agent, staging_bucket: str) -> SystemResource:
        """
        Deploy the full multi-agent tree as a single Vertex AI resource.

        Bundles local code (agents/, mock_data/, config.py) and a generated
        _vertex_env.py with current config values for the Vertex AI runtime.

        Args:
            root_agent: The root LlmAgent (Triage with all sub_agents).
            staging_bucket: GCS bucket gs://... for artifact staging.

        Returns:
            SystemResource with the resource_name and numeric_id of the created resource.
        """
        if not staging_bucket:
            raise RuntimeError(
                "VERTEX_STAGING_BUCKET not configured. "
                "Set VERTEX_STAGING_BUCKET=gs://your-bucket in .env to deploy the system."
            )

        self._init()
        import vertexai
        from vertexai import agent_engines
        from vertexai.agent_engines import AdkApp

        vertexai.init(
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.AGENT_ENGINE_LOCATION,
            staging_bucket=staging_bucket,
        )

        # CRITICAL: register local modules for by-value serialization BEFORE AdkApp.
        # Without this, cloudpickle serializes tool functions by module reference
        # (e.g. 'agents.specialists.booking.lookup_reservation') and the Vertex AI
        # runtime fails with 'No module named agents' when trying to import it.
        # With register_pickle_by_value the bytecode is inlined in the pickle.
        self._register_local_modules_for_pickle()

        adk_app = AdkApp(agent=root_agent, enable_tracing=True)
        logger.info("Deploying system '%s' to Vertex AI Agent Engine...", _SYSTEM_DISPLAY_NAME)

        tmp_dir = tempfile.mkdtemp(prefix="sfl_vertex_")
        try:
            extra_packages = self._get_extra_packages(tmp_dir)
            engine = agent_engines.create(
                agent_engine=adk_app,
                requirements=config.VERTEX_AGENT_REQUIREMENTS,
                extra_packages=extra_packages,
                display_name=_SYSTEM_DISPLAY_NAME,
                description=(
                    "Stayforlong multi-agent system: Triage + Booking + Support + "
                    "Property + HelpCenter. Managed via vertex_registry.py."
                ),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        rn = engine.resource_name
        logger.info("System deployed: %s", rn)
        return SystemResource(
            resource_name=rn,
            numeric_id=rn.split("/")[-1],
            display_name=_SYSTEM_DISPLAY_NAME,
        )

    def update_system(self, root_agent, staging_bucket: str) -> SystemResource:
        """
        Delete the existing resource and redeploy the system with the current version.

        Use when instructions, tools, or the model of any agent change.
        Equivalent to: --force in provision.py
        """
        existing = self.get_system()
        if existing:
            logger.info("Deleting existing system (%s)...", existing.resource_name)
            self._delete_resource(existing.resource_name)
        return self.deploy_system(root_agent, staging_bucket)

    def delete_system(self) -> bool:
        """
        Delete the system resource from Vertex AI.

        Returns:
            True if deleted, False if it did not exist.
        """
        existing = self.get_system()
        if not existing:
            logger.warning("System '%s' not found in Vertex AI.", _SYSTEM_DISPLAY_NAME)
            return False
        self._delete_resource(existing.resource_name)
        return True

    def purge_orphans(self) -> list[str]:
        """
        Delete orphan resources with prefix 'stayforlong-agent-' from a previous attempt.

        Returns:
            List of deleted resource_names.
        """
        self._init()
        from vertexai import agent_engines
        deleted = []
        try:
            for engine in agent_engines.list():
                dn = getattr(engine._gca_resource, "display_name", "") or ""
                if dn.startswith(_ORPHAN_PREFIX):
                    rn = engine.resource_name
                    logger.info("Deleting orphan '%s' (%s)...", dn, rn)
                    self._delete_resource(rn)
                    deleted.append(rn)
        except Exception:
            logger.exception("Error deleting orphan resources")
        return deleted

    # ── Internal ───────────────────────────────────────────────────────────────

    def _register_local_modules_for_pickle(self) -> None:
        """
        Register local modules for by-value serialization with cloudpickle.

        Problem without this method:
            cloudpickle serializes tool functions by module reference
            (e.g. 'agents.specialists.booking.lookup_reservation'). The Vertex AI
            runtime tries `import agents.specialists.booking` and fails with
            'No module named agents' even though they are in extra_packages.

        Solution:
            register_pickle_by_value(module) makes cloudpickle serialize the
            BYTECODE of the functions directly into the pickle (by value, not by ref).
            The Vertex AI runtime unpickles without needing to import anything.

        Reference:
            https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/troubleshooting/deploy
            "ensure all necessary dependencies used by the agent object are included"
        """
        import cloudpickle
        import importlib

        local_modules = [
            "agents",
            "agents.plugin",
            "agents.utils",
            "agents.constants",
            "agents.triage",
            "agents.specialists",
            "agents.specialists.booking",
            "agents.specialists.support",
            "agents.specialists.property",
            "agents.specialists.knowledge",
            "mock_data",
            "mock_data.reservations",
            "mock_data.incidents",
            "mock_data.properties",
            "config",
        ]
        for mod_name in local_modules:
            try:
                mod = importlib.import_module(mod_name)
                cloudpickle.register_pickle_by_value(mod)
                logger.debug("Registered for by-value pickle: %s", mod_name)
            except ImportError:
                logger.debug("Skip by-value pickle (not found): %s", mod_name)

        logger.info(
            "Registered %d local modules for by-value pickle.",
            len(local_modules),
        )

    def _init(self) -> None:
        if self._initialized:
            return
        if not config.GOOGLE_CLOUD_PROJECT:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured.")
        import vertexai
        vertexai.init(
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.AGENT_ENGINE_LOCATION,
        )
        self._initialized = True

    def _delete_resource(self, resource_name: str) -> None:
        from vertexai import agent_engines
        try:
            engine = agent_engines.get(resource_name)
            engine.delete(force=True)
            logger.info("Resource deleted: %s", resource_name)
        except Exception:
            logger.exception("Error deleting resource %s", resource_name)

    def _get_extra_packages(self, tmp_dir: str) -> list[str]:
        """
        Build the extra_packages list for agent_engines.create().

        Includes local code needed for the system to run on Vertex AI:
          - agents/      → utils, constants, plugin, triage, specialists/*
          - mock_data/   → test data (booking, support, property)
          - config.py    → configuration module (imports _vertex_env at runtime)
          - _vertex_env.py → generated with current config values
        """
        backend_dir = Path(__file__).parent.parent
        vertex_env_path = self._generate_vertex_env(tmp_dir)
        return [
            str(backend_dir / "agents"),
            str(backend_dir / "mock_data"),
            str(backend_dir / "config.py"),
            vertex_env_path,
        ]

    def _generate_vertex_env(self, tmp_dir: str) -> str:
        """
        Generate a _vertex_env.py with current config values baked in.

        This file is bundled in the artifact deployed to Vertex AI.
        config.py imports it (try/except ImportError) to inject project/location/model
        values into os.environ without needing a .env file at runtime.
        """
        lines = [
            "# Auto-generated by provision.py — DO NOT edit manually.",
            "# Injects environment variables into the Vertex AI Agent Engine runtime.",
            "import os",
            f"os.environ.setdefault('GOOGLE_CLOUD_PROJECT', {repr(config.GOOGLE_CLOUD_PROJECT)})",
            f"os.environ.setdefault('GOOGLE_CLOUD_LOCATION', {repr(config.AGENT_ENGINE_LOCATION)})",
            "os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', 'true')",
            f"os.environ.setdefault('GEMINI_MODEL', {repr(config.GEMINI_MODEL)})",
            f"os.environ.setdefault('VERTEX_AI_SEARCH_ENGINE_ID', {repr(config.VERTEX_AI_SEARCH_ENGINE_ID)})",
        ]
        path = os.path.join(tmp_dir, "_vertex_env.py")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.debug("_vertex_env.py generated at %s", path)
        return path
