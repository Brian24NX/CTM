"""TensorFlow-free, explicit time OR cumulative environment-step run budget."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RunBudget:
    seconds: int = 0
    steps: int = 0

    def __post_init__(self):
        if self.seconds < 0 or self.steps < 0 or bool(self.seconds) == bool(self.steps):
            raise ValueError('Specify exactly one positive duration or total environment-step target')
        if self.seconds > 3600:
            raise ValueError('Timed diagnostic is limited to 3600s; use --steps for full training')

    @classmethod
    def from_args(cls, seconds=None, steps=0):
        if seconds is None:
            seconds = 0 if steps else 3600
        return cls(seconds=seconds, steps=steps)

    def reached(self, elapsed_seconds, env_steps):
        return (env_steps >= self.steps) if self.steps else (elapsed_seconds >= self.seconds)

    @property
    def stop_reason(self):
        return 'environment_step_target' if self.steps else 'duration'
