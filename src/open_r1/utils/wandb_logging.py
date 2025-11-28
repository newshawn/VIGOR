import logging
import os
import subprocess
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group


def _flag_is_disabled(env_value: str) -> bool:
    return str(env_value).lower() in {"0", "false", "no", "off"}


def _normalize_report_to(report_to: Optional[Iterable[str]]) -> list[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [report_to]
    return list(report_to)


def save_git_patch_if_possible(report_to: Optional[Iterable[str]] = None) -> None:
    """
    Upload the current git diff to the active wandb run using wandb.save_git_patch().
    """
    if _flag_is_disabled(os.environ.get("INTUITOR_ENABLE_WANDB_GIT_PATCH", "1")):
        logger.info("INTUITOR_ENABLE_WANDB_GIT_PATCH is disabled; skip wandb.save_git_patch().")
        return

    normalized_report_to = {target.lower() for target in _normalize_report_to(report_to)}
    if "wandb" not in normalized_report_to:
        return

    try:
        import wandb
    except Exception:
        logger.warning("wandb not available; skip git patch upload.")
        return

    if wandb.run is None:
        logger.info("wandb.run not initialized yet; skip git patch upload.")
        return

    # Prefer native API if present; otherwise fall back to manual diff + wandb.save
    if hasattr(wandb, "save_git_patch"):
        try:
            wandb.save_git_patch()
            logger.info("Uploaded current git diff via wandb.save_git_patch().")
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to save git patch via wandb.save_git_patch: %s", exc)

    patch_path = _dump_git_diff(wandb.run.dir)
    if patch_path is None:
        return
    try:
        wandb.save(patch_path)
        logger.info("Uploaded git diff via wandb.save: %s", patch_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to upload git diff via wandb.save: %s", exc)


def _git_repo_root() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _dump_git_diff(run_dir: str) -> Optional[str]:
    repo_root = _git_repo_root()
    if not repo_root:
        logger.warning("Cannot locate git repository root; skip git diff upload.")
        return None

    try:
        diff_proc = subprocess.run(
            ["git", "-C", repo_root, "diff", "--binary", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        diff_content = diff_proc.stdout
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to compute git diff: %s", exc)
        return None

    if not diff_content.strip():
        logger.info("Git diff is empty; nothing to upload.")
        return None

    try:
        os.makedirs(run_dir, exist_ok=True)
        patch_path = os.path.join(run_dir, "git_diff.patch")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(diff_content)
        return patch_path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write git diff patch: %s", exc)
        return None
