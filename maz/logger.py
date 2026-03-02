"""Thin W&B logging wrapper. No-ops gracefully when wandb is absent."""


class Logger:
    """Logs training metrics to W&B if available, otherwise no-ops."""

    def __init__(self, use_wandb=False, project="maz", config=None):
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=project,
                    config=config or {},
                )
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False

    def log_selfplay(self, gen, games):
        """Log self-play statistics."""
        if not self.use_wandb:
            return
        lengths = [g.length for g in games]
        self._wandb.log({
            "selfplay/num_games": len(games),
            "selfplay/mean_length": sum(lengths) / max(len(lengths), 1),
            "selfplay/min_length": min(lengths) if lengths else 0,
            "selfplay/max_length": max(lengths) if lengths else 0,
        }, step=gen)

    def log_training(self, gen, metrics):
        """Log training metrics."""
        if not self.use_wandb or not metrics:
            return
        self._wandb.log({
            "train/value_loss": metrics["value_loss"],
            "train/policy_loss": metrics["policy_loss"],
            "train/total_loss": metrics["total_loss"],
            "train/num_positions": metrics["num_positions"],
        }, step=gen)

    def finish(self):
        """Finalize the W&B run."""
        if self.use_wandb and self._wandb:
            self._wandb.finish()
