"""
AgentLoader — descubre y construye agentes especialistas dinámicamente.

Escanea backend/agents/specialists/ buscando módulos que exporten
PLUGIN: AgentPlugin y construye LlmAgent objects a partir de ellos.

Uso en adk_runner.py:
    loader = AgentLoader()
    specialists, fallback = loader.build_agents()
    triage = build_triage_agent(specialists, fallback)

Uso en provision.py:
    loader = AgentLoader()
    plugins = loader.get_plugins()  # solo metadatos, sin construir LlmAgent
"""
import importlib
import pkgutil
import logging
from pathlib import Path
from google.adk.agents import LlmAgent
from agents.plugin import AgentPlugin

logger = logging.getLogger(__name__)


class AgentLoader:
    """Escanea agents/specialists/ y construye LlmAgent objects en startup."""

    _SPECIALISTS_PACKAGE = "agents.specialists"

    def __init__(self):
        self._plugins: list[AgentPlugin] = []
        self._loaded = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_agents(
        self,
    ) -> tuple[list[tuple[AgentPlugin, LlmAgent]], tuple[AgentPlugin, LlmAgent]]:
        """
        Construye y devuelve todos los agentes especialistas.

        Returns:
            (specialists, fallback) donde:
            - specialists: lista de (AgentPlugin, LlmAgent) para agentes no-fallback,
              en el orden en que aparecen en el filesystem.
            - fallback: (AgentPlugin, LlmAgent) del único agente con is_fallback=True.
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
        """Devuelve los metadatos crudos sin construir LlmAgent (para provision.py)."""
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
                        "Cargado plugin de agente: %s (fallback=%s)",
                        mod.PLUGIN.name,
                        mod.PLUGIN.is_fallback,
                    )
                else:
                    logger.warning(
                        "Módulo %s no tiene atributo PLUGIN: AgentPlugin — omitido.",
                        full_name,
                    )
            except Exception:
                logger.exception("Error al cargar plugin %s", full_name)

    def _validate(self) -> None:
        fallbacks = [p for p in self._plugins if p.is_fallback]
        if len(fallbacks) != 1:
            raise ValueError(
                f"Se esperaba exactamente 1 plugin con is_fallback=True, "
                f"encontrados {len(fallbacks)}: {[p.name for p in fallbacks]}. "
                f"Verifica agents/specialists/ — exactamente un módulo debe tener "
                f"PLUGIN = AgentPlugin(..., is_fallback=True)."
            )

        names = [p.name for p in self._plugins]
        seen: set[str] = set()
        duplicates = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(
                f"Nombres de agente duplicados en agents/specialists/: {duplicates}"
            )
