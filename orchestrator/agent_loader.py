"""
AgentLoader — dynamically discovers and builds specialist agents.

Scans agents/specialists/ for modules that export PLUGIN: AgentPlugin
and builds LlmAgent objects from them.

Usage in adk_runner.py:
    loader = AgentLoader()
    specialists, fallback = loader.build_agents()
    triage = build_triage_agent(specialists, fallback)

Usage in provision.py:
    loader = AgentLoader()
    plugins = loader.get_plugins()  # metadata only, no LlmAgent construction
"""
import importlib
import pkgutil
import logging
from pathlib import Path
from google.adk.agents import LlmAgent
from agents.plugin import AgentPlugin

logger = logging.getLogger(__name__)


class AgentLoader:
    """Scans agents/specialists/ and builds LlmAgent objects at startup."""

    _SPECIALISTS_PACKAGE = "agents.specialists"

    def __init__(self):
        self._plugins: list[AgentPlugin] = []
        self._loaded = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_agents(
        self,
    ) -> tuple[list[tuple[AgentPlugin, LlmAgent]], tuple[AgentPlugin, LlmAgent]]:
        """
        Build and return all specialist agents.

        Returns:
            (specialists, fallback) where:
            - specialists: list of (AgentPlugin, LlmAgent) for non-fallback agents,
              in filesystem discovery order.
            - fallback: (AgentPlugin, LlmAgent) for the single is_fallback=True agent.
        """
        self._ensure_loaded()
        specialists: list[tuple[AgentPlugin, LlmAgent]] = []
        fallback: tuple[AgentPlugin, LlmAgent] | None = None

        for plugin in self._plugins:
            agent = LlmAgent(
                name=plugin.name,
                model=plugin.model,
                instruction=plugin.instruction,
                tools=plugin.get_tools(),
            )
            if plugin.is_fallback:
                fallback = (plugin, agent)
            else:
                specialists.append((plugin, agent))

        return specialists, fallback  # type: ignore[return-value]

    def get_plugins(self) -> list[AgentPlugin]:
        """Return raw plugin metadata without building LlmAgent objects (used by provision.py)."""
        self._ensure_loaded()
        return list(self._plugins)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_plugins()
        self._validate()
        self._loaded = True

    def _load_plugins(self) -> None:
        package_path = Path(__file__).parent.parent / "agents" / "specialists"
        for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(package_path)]):
            full_name = f"{self._SPECIALISTS_PACKAGE}.{module_name}"
            try:
                mod = importlib.import_module(full_name)
                if hasattr(mod, "PLUGIN") and isinstance(mod.PLUGIN, AgentPlugin):
                    self._plugins.append(mod.PLUGIN)
                    logger.info(
                        "Loaded agent plugin: %s (fallback=%s)",
                        mod.PLUGIN.name,
                        mod.PLUGIN.is_fallback,
                    )
                else:
                    logger.warning(
                        "Module %s has no PLUGIN: AgentPlugin attribute — skipped.",
                        full_name,
                    )
            except Exception:
                logger.exception("Error loading plugin %s", full_name)

    def build_agents_merged(
        self,
    ) -> tuple[list[tuple[AgentPlugin, LlmAgent]], tuple[AgentPlugin, LlmAgent]]:
        """
        Build agents merging Python source definitions with GCS-stored configs.

        Resolution order:
        1. Python source agents are loaded as the base.
        2. GCS entries with the same name override instruction/routing_hint/model/tools.
        3. GCS entries with new names are appended as additional agents.

        Returns:
            (specialists, fallback) same shape as build_agents().
        """
        from services.agent_gcs_store import load_all
        from agents.tool_registry import get_tools_for

        self._ensure_loaded()
        gcs_configs = load_all()

        specialists: list[tuple[AgentPlugin, LlmAgent]] = []
        fallback: tuple[AgentPlugin, LlmAgent] | None = None
        processed_names: set[str] = set()

        # Python source agents — apply GCS overrides where present
        for plugin in self._plugins:
            processed_names.add(plugin.name)
            override = gcs_configs.get(plugin.name, {})

            instruction = override.get("instruction", plugin.instruction)
            routing_hint = override.get("routing_hint", plugin.routing_hint)
            model = override.get("model", plugin.model)
            is_fallback = override.get("is_fallback", plugin.is_fallback)

            if "tools" in override:
                tools = get_tools_for(override["tools"])
            else:
                tools = plugin.get_tools()

            agent = LlmAgent(
                name=plugin.name,
                model=model,
                instruction=instruction,
                tools=tools,
            )
            if is_fallback:
                fallback = (plugin, agent)
            else:
                specialists.append((plugin, agent))

        # GCS-only agents (names not present in Python source files)
        for name, cfg in gcs_configs.items():
            if name in processed_names:
                continue
            tools = get_tools_for(cfg.get("tools", ["transfer_to_triage"]))
            synthetic_plugin = AgentPlugin(
                name=cfg["name"],
                routing_hint=cfg.get("routing_hint", ""),
                instruction=cfg.get("instruction", ""),
                model=cfg.get("model", "gemini-2.5-flash"),
                is_fallback=cfg.get("is_fallback", False),
                get_tools=lambda t=tools: t,
            )
            agent = LlmAgent(
                name=synthetic_plugin.name,
                model=synthetic_plugin.model,
                instruction=synthetic_plugin.instruction,
                tools=tools,
            )
            if synthetic_plugin.is_fallback:
                fallback = (synthetic_plugin, agent)
            else:
                specialists.append((synthetic_plugin, agent))

        return specialists, fallback  # type: ignore[return-value]

    def _validate(self) -> None:
        fallbacks = [p for p in self._plugins if p.is_fallback]
        if len(fallbacks) != 1:
            raise ValueError(
                f"Expected exactly 1 plugin with is_fallback=True, "
                f"found {len(fallbacks)}: {[p.name for p in fallbacks]}. "
                f"Check agents/specialists/ — exactly one module must have "
                f"PLUGIN = AgentPlugin(..., is_fallback=True)."
            )

        names = [p.name for p in self._plugins]
        seen: set[str] = set()
        duplicates = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(
                f"Duplicate agent names in agents/specialists/: {duplicates}"
            )
