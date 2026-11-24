import math
from dataclasses import asdict, dataclass

from config import TokenSuperpositionConfig


@dataclass
class SuperpositionState:
    best_loss: float | None = None
    stale_validations: int = 0
    recovery_step: int | None = None


class SuperpositionSchedule:
    def __init__(self, config: TokenSuperpositionConfig):
        self.config = config
        self.state = SuperpositionState()

    def _switch(self, step, reason):
        self.state.recovery_step = step
        return f'switch to next-token training at step {step}; reason={reason}'

    def prepare_step(self, step):
        if self.state.recovery_step is not None:
            return None
        if self.config.mode == 'fixed' and step >= self.config.switch_step:
            return self._switch(step, 'configured switch step')
        if self.config.mode == 'adaptive' and step >= self.config.max_superposition_steps:
            return self._switch(step, 'maximum superposition steps')
        return None

    @property
    def bag_size(self):
        if self.config.mode == 'standard' or self.state.recovery_step is not None:
            return 1
        return self.config.bag_size

    def observe_validation(self, next_step, loss):
        if self.config.mode != 'adaptive' or self.bag_size == 1:
            return None
        if not math.isfinite(loss):
            raise ValueError(f'Adaptive TST requires a finite validation loss, got {loss}')

        state = self.state
        improved = state.best_loss is None or (
            loss < state.best_loss and state.best_loss - loss >= self.config.min_delta
        )
        if improved:
            state.best_loss = loss
            state.stale_validations = 0
        elif next_step >= self.config.min_superposition_steps:
            state.stale_validations += 1

        status = (
            f'next_step={next_step} val_loss={loss:.6f} best={state.best_loss:.6f} '
            f'stale={state.stale_validations}/{self.config.patience}'
        )
        if state.stale_validations >= self.config.patience:
            return f'{status}; {self._switch(next_step, "validation loss plateau")}'
        return f'{status}; continue superposition'

    def state_dict(self):
        return {'config': self.config.model_dump(), 'state': asdict(self.state)}

    def load_state_dict(self, saved, *, reset_validation_history=False):
        if saved['config'] != self.config.model_dump():
            raise ValueError('Token superposition settings must match the checkpoint when resuming TST')
        self.state = SuperpositionState(**saved['state'])
        if reset_validation_history:
            self.state.best_loss = None
            self.state.stale_validations = 0
