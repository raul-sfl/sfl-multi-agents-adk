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
