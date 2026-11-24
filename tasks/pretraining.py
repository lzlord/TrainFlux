# Modified by TrainFlux.

import torch

from tasks.base import TaskStepOutput
from tasks.causal import CausalTask


def next_bag_cross_entropy(logits, targets):
    # Reuse the normalizer for every occurrence, including repeated target IDs.
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    return (logits.logsumexp(dim=-1) - logits.gather(-1, targets).mean(dim=-1)).mean()


class PretrainingTask(CausalTask):
    name: str = 'pretraining'

    def train_micro_step(self, model, batch, assets):
        x, y, _ = batch
        if x.ndim == 2:
            return super().train_micro_step(model, batch, assets)

        device = self.ctx.device
        precision = self.ctx.precision
        x = x.to(device.device, non_blocking=True)
        y = y.to(device.device, non_blocking=True)
        with torch.autocast(
            device_type=device.device_type,
            dtype=precision.autocast_dtype,
            enabled=precision.use_autocast,
        ):
            loss = next_bag_cross_entropy(model(x)['logits'], y)

        n_valid = torch.tensor(y.numel(), device=y.device)
        return TaskStepOutput(
            tokens_processed=x.numel(),
            n_valid=n_valid,
            loss=loss,
            loss_for_backward=loss * n_valid,
            metrics={'Train Loss': loss.detach(), 'Train MCE Loss': loss.detach()},
        )
