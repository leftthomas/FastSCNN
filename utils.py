import warnings

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRScheduler(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps (int): after this step, we stop decreasing learning rate.
        power (float): The power of the polynomial.
            Default: 0.9.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_decay_steps, power=0.9, last_epoch=-1):
        self.max_decay_steps = max_decay_steps
        self.power = power
        super(PolynomialLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (self.last_epoch / self.max_decay_steps) ** self.power
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.last_epoch / self.max_decay_steps) ** self.power
                for base_lr in self.base_lrs]
