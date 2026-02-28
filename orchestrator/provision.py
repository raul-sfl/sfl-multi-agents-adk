"""
provision.py — Gestiona el sistema multi-agente en Vertex AI Agent Engine.

Despliega el árbol completo (Triage + todos los especialistas) como un único
Vertex AI Reasoning Engine resource siguiendo el patrón ADK estándar.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMANDOS (ejecutar desde backend/):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Deploy del sistema si no existe
  python -m orchestrator.provision

  # Forzar redeploy (actualiza instrucciones, tools o modelo)
  python -m orchestrator.provision --force

  # Ver estado actual en Vertex AI
  python -m orchestrator.provision --list

  # Eliminar el recurso del sistema
  python -m orchestrator.provision --delete

  # Limpiar recursos huérfanos de la tentativa anterior (stayforlong-agent-*)
  python -m orchestrator.provision --purge-orphans

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREREQUISITOS:
  GOOGLE_CLOUD_PROJECT=...
  VERTEX_STAGING_BUCKET=gs://tu-bucket
  GOOGLE_GENAI_USE_VERTEXAI=true

STARTUP AUTOMÁTICO (main.py):
  Llamado en background al arrancar — crea el recurso si no existe.
  Si ya está desplegado, termina en ~2s sin hacer nada.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import sys
import logging
import config
from orchestrator.vertex_registry import VertexRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Guard de configuración ─────────────────────────────────────────────────────

def _check_config(need_bucket: bool = True) -> bool:
    if not config.GOOGLE_CLOUD_PROJECT:
        logger.warning("GOOGLE_CLOUD_PROJECT no configurado — saltando aprovisionamiento.")
        return False
    if need_bucket and not config.VERTEX_STAGING_BUCKET:
        logger.warning(
            "VERTEX_STAGING_BUCKET no configurado — saltando aprovisionamiento. "
            "Establece VERTEX_STAGING_BUCKET=gs://tu-bucket en .env."
        )
        return False
    return True


# ── Comandos ───────────────────────────────────────────────────────────────────

def cmd_list() -> None:
    """Muestra el estado actual del sistema en Vertex AI."""
    if not _check_config(need_bucket=False):
        return

    registry = VertexRegistry()
    all_resources = registry.list_all()
    system = registry.get_system()

    print("\nVertex AI Agent Engine — Stayforlong Multi-Agent System")
    print("=" * 70)

    if system:
        print(f"\n  ✅ SISTEMA ACTIVO")
        print(f"     Nombre:   {system.display_name}")
        print(f"     ID:       {system.numeric_id}")
        print(f"     Resource: {system.resource_name}")
        print(f"\n  Agentes en el árbol:")
        # Importar root_agent para listar sub_agents
        try:
            from agent import root_agent
            print(f"     • {root_agent.name} (router principal)")
            for sub in getattr(root_agent, 'sub_agents', []):
                print(f"       └─ {sub.name}")
        except Exception:
            print("     (no se pudo leer el árbol de agentes locales)")
    else:
        print(f"\n  ❌ Sistema '{system or 'stayforlong-multiagent'}' NO desplegado")

    orphans = [r for r in all_resources if r.display_name.startswith("stayforlong-agent-")]
    if orphans:
        print(f"\n  ⚠️  Recursos huérfanos detectados ({len(orphans)}):")
        for o in orphans:
            print(f"     • {o.display_name} ({o.numeric_id}) — ejecuta --purge-orphans para eliminarlos")

    print()
    if system:
        print(f"  Vertex AI console:")
        print(f"  https://console.cloud.google.com/vertex-ai/agents?project={config.GOOGLE_CLOUD_PROJECT}")
        print(f"\n  Cloud Trace (trazas por agente):")
        print(f"  https://console.cloud.google.com/traces/list?project={config.GOOGLE_CLOUD_PROJECT}")
    print()


def cmd_purge_orphans() -> None:
    """Elimina recursos huérfanos con prefijo stayforlong-agent-."""
    if not _check_config(need_bucket=False):
        return

    registry = VertexRegistry()
    print("\nEliminando recursos huérfanos (stayforlong-agent-*)...")
    deleted = registry.purge_orphans()
    if deleted:
        print(f"\n✅ {len(deleted)} recurso(s) eliminado(s):")
        for rn in deleted:
            print(f"   {rn}")
    else:
        print("\n✅ No se encontraron recursos huérfanos.")
    print()


def cmd_delete() -> None:
    """Elimina el recurso del sistema de Vertex AI."""
    if not _check_config(need_bucket=False):
        return

    registry = VertexRegistry()
    deleted = registry.delete_system()
    if deleted:
        print("\n✅ Sistema eliminado de Vertex AI.")
        print("   Recuerda borrar o comentar AGENT_ENGINE_ID en .env si vas a usar InMemory.\n")
    else:
        print("\n⚠️  Sistema no encontrado en Vertex AI.\n")


def run_provision(force: bool = False) -> None:
    """
    Flujo principal de aprovisionamiento (usado desde startup de la app y CLI).

    Args:
        force: Si True, redespliega aunque ya exista (--force).
               Si False (defecto), sólo despliega si no existe.
    """
    if not _check_config():
        return

    # Importar root_agent desde el entrypoint estándar ADK
    from agent import root_agent

    registry = VertexRegistry()

    if not force:
        existing = registry.get_system()
        if existing:
            logger.info(
                "Sistema '%s' ya desplegado (%s) — nada que hacer.",
                existing.display_name,
                existing.numeric_id,
            )
            return

    action = "Actualizando" if force else "Desplegando"
    logger.info("%s sistema multi-agente en Vertex AI...", action)

    try:
        if force:
            resource = registry.update_system(root_agent, config.VERTEX_STAGING_BUCKET)
        else:
            resource = registry.deploy_system(root_agent, config.VERTEX_STAGING_BUCKET)
    except Exception:
        logger.exception("Error al desplegar el sistema en Vertex AI")
        return

    _print_summary(resource)


def _print_summary(resource) -> None:
    print("\nVertex AI Agent Engine — Sistema desplegado")
    print("=" * 70)
    print(f"  ✅ {resource.display_name}")
    print(f"     ID:       {resource.numeric_id}")
    print(f"     Resource: {resource.resource_name}")
    print(f"\n  Actualiza tu .env con:")
    print(f"     AGENT_ENGINE_ID={resource.numeric_id}")
    print(f"\n  Vertex AI console:")
    print(f"  https://console.cloud.google.com/vertex-ai/agents?project={config.GOOGLE_CLOUD_PROJECT}")
    print(f"\n  Cloud Trace (trazas por agente):")
    print(f"  https://console.cloud.google.com/traces/list?project={config.GOOGLE_CLOUD_PROJECT}")
    print()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gestiona el sistema multi-agente Stayforlong en Vertex AI Agent Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python -m orchestrator.provision                  # deploy si no existe\n"
            "  python -m orchestrator.provision --force          # redeploy completo\n"
            "  python -m orchestrator.provision --list           # ver estado actual\n"
            "  python -m orchestrator.provision --delete         # eliminar recurso\n"
            "  python -m orchestrator.provision --purge-orphans  # limpiar tentativa anterior\n"
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="Muestra el estado actual del sistema en Vertex AI.",
    )
    group.add_argument(
        "--force", "-f",
        action="store_true",
        help="Redespliega el sistema completo (actualiza instrucciones/tools/modelo).",
    )
    group.add_argument(
        "--delete", "-d",
        action="store_true",
        help="Elimina el recurso del sistema de Vertex AI.",
    )
    group.add_argument(
        "--purge-orphans",
        action="store_true",
        help="Elimina recursos huérfanos (stayforlong-agent-*) de tentativas anteriores.",
    )

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.delete:
        cmd_delete()
    elif args.purge_orphans:
        cmd_purge_orphans()
    elif args.force:
        run_provision(force=True)
    else:
        run_provision(force=False)


if __name__ == "__main__":
    main()
