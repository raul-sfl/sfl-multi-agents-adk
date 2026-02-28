"""
provision.py — Manages the multi-agent system on Vertex AI Agent Engine.

Deploys the full agent tree (Triage + all specialists) as a single
Vertex AI Reasoning Engine resource following the standard ADK pattern.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMANDS (run from the repo root):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Deploy the system if it does not exist
  python -m orchestrator.provision

  # Force redeploy (updates instructions, tools or model)
  python -m orchestrator.provision --force

  # Show current state in Vertex AI
  python -m orchestrator.provision --list

  # Delete the system resource
  python -m orchestrator.provision --delete

  # Remove orphan resources from a previous attempt (stayforlong-agent-*)
  python -m orchestrator.provision --purge-orphans

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREREQUISITES:
  GOOGLE_CLOUD_PROJECT=...
  VERTEX_STAGING_BUCKET=gs://your-bucket
  GOOGLE_GENAI_USE_VERTEXAI=true

AUTO-STARTUP (main.py):
  Called in the background on server start — creates the resource if it does not exist.
  If already deployed, completes in ~2s with no action.
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


# ── Config guard ──────────────────────────────────────────────────────────

def _check_config(need_bucket: bool = True) -> bool:
    if not config.GOOGLE_CLOUD_PROJECT:
        logger.warning("GOOGLE_CLOUD_PROJECT not set — skipping provisioning.")
        return False
    if need_bucket and not config.VERTEX_STAGING_BUCKET:
        logger.warning(
            "VERTEX_STAGING_BUCKET not set — skipping provisioning. "
            "Set VERTEX_STAGING_BUCKET=gs://your-bucket in .env."
        )
        return False
    return True


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_list() -> None:
    """Show current state of the system in Vertex AI."""
    if not _check_config(need_bucket=False):
        return

    registry = VertexRegistry()
    all_resources = registry.list_all()
    system = registry.get_system()

    print("\nVertex AI Agent Engine — Stayforlong Multi-Agent System")
    print("=" * 70)

    if system:
        print(f"\n  ✅ SYSTEM ACTIVE")
        print(f"     Name:     {system.display_name}")
        print(f"     ID:       {system.numeric_id}")
        print(f"     Resource: {system.resource_name}")
        print(f"\n  Agents in the tree:")
        try:
            from agent import root_agent
            print(f"     • {root_agent.name} (main router)")
            for sub in getattr(root_agent, 'sub_agents', []):
                print(f"       └─ {sub.name}")
        except Exception:
            print("     (could not read local agent tree)")
    else:
        print(f"\n  ❌ System '{system or 'stayforlong-multiagent'}' NOT deployed")

    orphans = [r for r in all_resources if r.display_name.startswith("stayforlong-agent-")]
    if orphans:
        print(f"\n  ⚠️  Orphan resources detected ({len(orphans)}):")
        for o in orphans:
            print(f"     • {o.display_name} ({o.numeric_id}) — run --purge-orphans to delete them")

    print()
    if system:
        print(f"  Vertex AI console:")
        print(f"  https://console.cloud.google.com/vertex-ai/agents?project={config.GOOGLE_CLOUD_PROJECT}")
        print(f"\n  Cloud Trace (per-agent traces):")
        print(f"  https://console.cloud.google.com/traces/list?project={config.GOOGLE_CLOUD_PROJECT}")
    print()


def cmd_purge_orphans() -> None:
    """Delete orphan resources with prefix stayforlong-agent-."""
    if not _check_config(need_bucket=False):
        return

    registry = VertexRegistry()
    print("\nDeleting orphan resources (stayforlong-agent-*)...")
    deleted = registry.purge_orphans()
    if deleted:
        print(f"\n✅ {len(deleted)} resource(s) deleted:")
        for rn in deleted:
            print(f"   {rn}")
    else:
        print("\n✅ No orphan resources found.")
    print()


def cmd_delete() -> None:
    """Delete the system resource from Vertex AI."""
    if not _check_config(need_bucket=False):
        return

    registry = VertexRegistry()
    deleted = registry.delete_system()
    if deleted:
        print("\n✅ System deleted from Vertex AI.")
        print("   Remember to clear or comment out AGENT_ENGINE_ID in .env if switching to InMemory.\n")
    else:
        print("\n⚠️  System not found in Vertex AI.\n")


def run_provision(force: bool = False) -> None:
    """
    Main provisioning flow (called from app startup and CLI).

    Args:
        force: If True, redeploys even if the system already exists (--force).
               If False (default), only deploys if the system does not exist.
    """
    if not _check_config():
        return

    from agent import root_agent

    registry = VertexRegistry()

    if not force:
        existing = registry.get_system()
        if existing:
            logger.info(
                "System '%s' already deployed (%s) — nothing to do.",
                existing.display_name,
                existing.numeric_id,
            )
            return

    action = "Updating" if force else "Deploying"
    logger.info("%s multi-agent system on Vertex AI...", action)

    try:
        if force:
            resource = registry.update_system(root_agent, config.VERTEX_STAGING_BUCKET)
        else:
            resource = registry.deploy_system(root_agent, config.VERTEX_STAGING_BUCKET)
    except Exception:
        logger.exception("Error deploying system to Vertex AI")
        return

    _print_summary(resource)


def _print_summary(resource) -> None:
    print("\nVertex AI Agent Engine — System deployed")
    print("=" * 70)
    print(f"  ✅ {resource.display_name}")
    print(f"     ID:       {resource.numeric_id}")
    print(f"     Resource: {resource.resource_name}")
    print(f"\n  Update your .env with:")
    print(f"     AGENT_ENGINE_ID={resource.numeric_id}")
    print(f"\n  Vertex AI console:")
    print(f"  https://console.cloud.google.com/vertex-ai/agents?project={config.GOOGLE_CLOUD_PROJECT}")
    print(f"\n  Cloud Trace (per-agent traces):")
    print(f"  https://console.cloud.google.com/traces/list?project={config.GOOGLE_CLOUD_PROJECT}")
    print()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the Stayforlong multi-agent system on Vertex AI Agent Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m orchestrator.provision                  # deploy if not exists\n"
            "  python -m orchestrator.provision --force          # full redeploy\n"
            "  python -m orchestrator.provision --list           # show current state\n"
            "  python -m orchestrator.provision --delete         # delete resource\n"
            "  python -m orchestrator.provision --purge-orphans  # clean up previous attempt\n"
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="Show current state of the system in Vertex AI.",
    )
    group.add_argument(
        "--force", "-f",
        action="store_true",
        help="Redeploy the full system (updates instructions/tools/model).",
    )
    group.add_argument(
        "--delete", "-d",
        action="store_true",
        help="Delete the system resource from Vertex AI.",
    )
    group.add_argument(
        "--purge-orphans",
        action="store_true",
        help="Delete orphan resources (stayforlong-agent-*) from previous attempts.",
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
