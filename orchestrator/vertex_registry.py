"""
VertexRegistry — gestiona el sistema multi-agente en Vertex AI Agent Engine.

El sistema completo (Triage + todos los especialistas) se despliega como UN ÚNICO
Vertex AI Reasoning Engine resource. Esto sigue el patrón ADK estándar:
  - root_agent (con sub_agents) → AdkApp → agent_engines.create()

Beneficios vs. deploy por agente separado:
  - El routing inter-agente funciona (transfer_to_agent opera dentro del árbol)
  - Un solo recurso visible en Vertex AI console con el sistema completo
  - Cloud Trace muestra trazas por agente individual dentro de cada conversación
  - Más robusto: los agentes no son entidades aisladas sin contexto
  - Sigue la convención ADK estándar

Nombre del recurso: "stayforlong-multiagent"

CLI:
    cd backend
    python -m orchestrator.provision             # deploy si no existe
    python -m orchestrator.provision --force     # redeploy (update instrucciones)
    python -m orchestrator.provision --list      # ver estado actual
    python -m orchestrator.provision --delete    # eliminar recurso
    python -m orchestrator.provision --purge-orphans  # limpiar recursos anteriores
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
_ORPHAN_PREFIX = "stayforlong-agent-"  # prefijo de recursos de la tentativa anterior


@dataclass
class SystemResource:
    resource_name: str  # path completo: projects/P/locations/L/reasoningEngines/N
    numeric_id: str     # solo el número: "1234567890123456"
    display_name: str


class VertexRegistry:
    """
    Gestiona el recurso Vertex AI Reasoning Engine del sistema multi-agente.

    El árbol completo (root_agent con todos los sub_agents) se despliega como
    un único recurso llamado "stayforlong-multiagent".

    El código local (agents/, mock_data/, config.py) se bundlea como extra_packages
    junto con _vertex_env.py generado que inyecta los valores de config en el runtime.
    """

    def __init__(self):
        self._initialized = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_system(self) -> Optional[SystemResource]:
        """Devuelve el recurso del sistema si está desplegado, o None."""
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
            logger.exception("Error al buscar el sistema en Vertex AI")
        return None

    def list_all(self) -> list[SystemResource]:
        """Lista TODOS los recursos stayforlong-* (sistema + posibles huérfanos)."""
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
            logger.exception("Error al listar recursos Vertex AI")
        return results

    def deploy_system(self, root_agent, staging_bucket: str) -> SystemResource:
        """
        Despliega el árbol multi-agente completo como un único Vertex AI resource.

        Bundlea el código local (agents/, mock_data/, config.py) y un _vertex_env.py
        generado con los valores de config actuales para el runtime de Vertex AI.

        Args:
            root_agent: El LlmAgent raíz (Triage con todos los sub_agents).
            staging_bucket: GCS bucket gs://... para staging del artefacto.

        Returns:
            SystemResource con resource_name y numeric_id del recurso creado.
        """
        if not staging_bucket:
            raise RuntimeError(
                "VERTEX_STAGING_BUCKET no configurado. "
                "Establece gs://tu-bucket en .env para desplegar el sistema."
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

        # CRÍTICO: registrar módulos locales para serialización by-value ANTES de AdkApp.
        # Sin esto, cloudpickle serializa las tool functions por referencia al módulo
        # ('agents.specialists.booking.lookup_reservation') y el runtime de Vertex AI
        # falla con 'No module named agents' al intentar importarlo.
        # Con register_pickle_by_value el bytecode queda inline en el pickle.
        self._register_local_modules_for_pickle()

        adk_app = AdkApp(agent=root_agent, enable_tracing=True)
        logger.info("Desplegando sistema '%s' en Vertex AI Agent Engine...", _SYSTEM_DISPLAY_NAME)

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
                    "Alojamientos + HelpCenter. Managed via vertex_registry.py."
                ),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        rn = engine.resource_name
        logger.info("Sistema desplegado: %s", rn)
        return SystemResource(
            resource_name=rn,
            numeric_id=rn.split("/")[-1],
            display_name=_SYSTEM_DISPLAY_NAME,
        )

    def update_system(self, root_agent, staging_bucket: str) -> SystemResource:
        """
        Elimina el recurso existente y redespliega el sistema con la versión actual.

        Usar cuando cambian instrucciones, tools o el modelo de cualquier agente.
        Equivale a: --force en provision.py
        """
        existing = self.get_system()
        if existing:
            logger.info("Eliminando sistema anterior (%s)...", existing.resource_name)
            self._delete_resource(existing.resource_name)
        return self.deploy_system(root_agent, staging_bucket)

    def delete_system(self) -> bool:
        """
        Elimina el recurso del sistema de Vertex AI.

        Returns:
            True si se eliminó, False si no existía.
        """
        existing = self.get_system()
        if not existing:
            logger.warning("Sistema '%s' no encontrado en Vertex AI.", _SYSTEM_DISPLAY_NAME)
            return False
        self._delete_resource(existing.resource_name)
        return True

    def purge_orphans(self) -> list[str]:
        """
        Elimina recursos huérfanos con el prefijo 'stayforlong-agent-' (tentativa anterior).

        Returns:
            Lista de resource_names eliminados.
        """
        self._init()
        from vertexai import agent_engines
        deleted = []
        try:
            for engine in agent_engines.list():
                dn = getattr(engine._gca_resource, "display_name", "") or ""
                if dn.startswith(_ORPHAN_PREFIX):
                    rn = engine.resource_name
                    logger.info("Eliminando huérfano '%s' (%s)...", dn, rn)
                    self._delete_resource(rn)
                    deleted.append(rn)
        except Exception:
            logger.exception("Error al eliminar recursos huérfanos")
        return deleted

    # ── Internal ───────────────────────────────────────────────────────────────

    def _register_local_modules_for_pickle(self) -> None:
        """
        Registra módulos locales para serialización by-value con cloudpickle.

        Problema sin este método:
            cloudpickle serializa las tool functions por referencia a su módulo
            (ej. 'agents.specialists.booking.lookup_reservation'). El runtime de
            Vertex AI intenta `import agents.specialists.booking` y falla con
            'No module named agents' aunque estén en extra_packages.

        Solución:
            register_pickle_by_value(module) hace que cloudpickle serialice el
            BYTECODE de las funciones directamente en el pickle (by value, no by ref).
            El runtime de Vertex AI desempaqueta el pickle sin necesitar importar nada.

        Referencia:
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
            "Módulos locales registrados para by-value pickle: %d módulos",
            len(local_modules),
        )

    def _init(self) -> None:
        if self._initialized:
            return
        if not config.GOOGLE_CLOUD_PROJECT:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT no está configurado.")
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
            logger.info("Recurso eliminado: %s", resource_name)
        except Exception:
            logger.exception("Error al eliminar recurso %s", resource_name)

    def _get_extra_packages(self, tmp_dir: str) -> list[str]:
        """
        Construye la lista de extra_packages para agent_engines.create().

        Incluye el código local necesario para que el sistema funcione en Vertex AI:
          - agents/      → utils, constants, plugin, triage, specialists/*
          - mock_data/   → datos de prueba (booking, support, property)
          - config.py    → módulo de configuración (importa _vertex_env en runtime)
          - _vertex_env.py → generado con valores actuales de config
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
        Genera un _vertex_env.py con los valores de config actuales baked-in.

        Este archivo se bundlea en el artefacto desplegado en Vertex AI.
        config.py lo importa (try/except ImportError) para inyectar en os.environ
        los valores de proyecto/location/model sin necesitar .env en el runtime.
        """
        lines = [
            "# Auto-generado por provision.py — NO editar manualmente.",
            "# Inyecta variables de entorno en el runtime de Vertex AI Agent Engine.",
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
        logger.debug("_vertex_env.py generado en %s", path)
        return path
