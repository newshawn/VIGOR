import os


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    # Default to offline mode for air-gapped / restricted environments.
    # Users can still override this by exporting WANDB_MODE=online before launching.
    os.environ.setdefault("WANDB_MODE", "offline")
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
