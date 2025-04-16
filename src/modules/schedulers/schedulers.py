import warnings
import math
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler): 
    
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr = 0.0,
        eta_min = 0.0,
        last_epoch = -1,
    ):
        
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn(
            "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
            UserWarning,
        )

        if self.last_epoch == self.warmup_steps:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_steps:
            return [ 
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1) for base_lr,
                group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_steps) % (2 * (self.max_steps - self.warmup_steps)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_steps - self.warmup_steps))) / 2 for base_lr,
                group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps))) / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps - 1) / (self.max_steps - self.warmup_steps))) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]


    def _get_closed_form_lr(self):
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_steps - 1) for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps))) for base_lr in self.base_lrs
        ]
