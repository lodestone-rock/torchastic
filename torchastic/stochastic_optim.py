import warnings

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class Compass(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.99, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 5).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        reset_every (int, optional):
            Reset the optimizer state every this many steps. If None, never reset.
        cautious (bool):
            Whether to apply cautious gradient updates (default: True).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.99, 0.999),
        amp_fac=5,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
        reset_every=None,
        cautious=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            reset_every=reset_every,
            cautious=cautious,
        )
        super(Compass, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            reset_every = group["reset_every"]
            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["last_reset_step"] = state["step"]
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16
                    )

                # unpack
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema = state["ema"].to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                steps_since_reset = (state["step"] - state["last_reset_step"]) + 1
                state["step"] += 1

                # Auto-reset if configured
                if reset_every is not None and state["step"] % reset_every == 0:
                    self.reset_state()

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # bias correction step size using steps since last reset
                bias_correction = 1 - beta1**steps_since_reset
                bias_correction_sqrt = (1 - beta2**steps_since_reset) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                # Compute the proposed update
                if group["cautious"]:
                    update = -step_size * (ema / denom)
                    # Check if gradient agrees with update direction
                    caution_mask = (update * grad > 0).to(grad.dtype)

                    # Scale gradient where directions agree
                    grad = grad * (
                        1
                        - caution_mask
                        + caution_mask / (grad.abs().mean() + group["eps"])
                    )

                # Apply the update with cautious gradient
                p_fp32.data.addcdiv_(ema, denom, value=-step_size)

                # pack
                copy_stochastic_(state["ema"], ema)
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(p, p_fp32)

        return loss

    def reset_state(self):
        """Resets the optimizer state (step count, momentum, and adaptive
        learning rate history).  Can be beneficial to call periodically (e.g.,
        every 1000 steps) to clear accumulated traces for adaptive optimizers."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) != 0:  # Only reset if state exists
                    state["last_reset_step"] = state["step"]  # Store when we last reset
                    state["ema"].zero_()
                    state["ema_squared"].zero_()


class LPFAdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.9, 0.999),
        amp_fac=2,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(LPFAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["smoothing"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16
                    )

                # unpack
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                smoothing = state["smoothing"].to(torch.float32)
                ema = state["ema"].to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)

                beta1, beta2, beta3 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta2 ** state["step"]
                bias_correction_sqrt = (1 - beta3 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta2) * grad
                smoothing.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(smoothing, alpha=amplification_factor)

                ema.mul_(beta2).add_(grad, alpha=1 - beta2)
                # ema_squared = ema + (1 - beta3) * grad ** 2
                ema_squared.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(ema, denom, value=-step_size)

                # pack
                copy_stochastic_(state["ema"], ema)
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(state["smoothing"], smoothing)
                copy_stochastic_(p, p_fp32)

        return loss


class AdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16
                    )

                # unpack
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema = state["ema"].to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad

                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(ema, denom, value=-step_size)

                # pack
                copy_stochastic_(state["ema"], ema)
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(p, p_fp32)

        return loss


class RMSProp(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        beta (float, optional):
            coefficients used for computing running averages of
            gradient and its square (default: 0.999).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta=0.999,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16
                    )

                # unpack
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)

                beta = group["beta"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # bias correction step size
                bias_correction_sqrt = (1 - beta ** state["step"]) ** (1 / 2)
                step_size = lr

                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-step_size)

                # pack
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(p, p_fp32)

        return loss


class StochasticAccumulator:
    """
    # init your model
    your_fancy_model = YourFancyModel(*your_model_args)

    # apply stochastic grad accumulator hooks
    StochasticAccumulator.assign_hooks(your_fancy_model)

    # training
    while True:
        loss = your_fancy_model.loss(*your_model_input)
        for _ in range(grad_accum_length):
            loss.backward()

        # apply grad buffer back
        StochasticAccumulator.reassign_grad_buffer(your_fancy_model)

        optimizer.step()
        optimizer.zero_grad()
    """

    @staticmethod
    def stochastic_grad_accum(p):
        # hack by adding attributes to "grad"
        if hasattr(p, "acc_grad"):
            acc_grad_fp32 = p.acc_grad.clone().to(torch.float32)
            # acc_grad_fp32 += fp_32_grad
            # upcast the gradient and then add it to p.grad
            acc_grad_fp32.add_(p.grad.to(torch.float32))
            copy_stochastic_(p.acc_grad, acc_grad_fp32)
            del acc_grad_fp32
            del p.grad
        else:
            p.acc_grad = p.grad.clone().to(torch.bfloat16)
            del p.grad

    @staticmethod
    def reassign_grad_buffer(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.grad = p.acc_grad
                del p.acc_grad

    @staticmethod
    def assign_hooks(model):
        hooks = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                hook = p.register_post_accumulate_grad_hook(
                    StochasticAccumulator.stochastic_grad_accum
                )
                hooks.append(hook)
        return hooks


class ResetWrapper(_LRScheduler):
    """Wraps a base scheduler to add periodic linear warmups.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of steps for each warmup period
        reset_every: Reset and warmup every this many steps
        base_scheduler: Base learning rate scheduler to wrap
        last_epoch: The index of last epoch (-1 by default)

    Example:
        >>> base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        >>> scheduler = ResetWrapper(
        ...     optimizer,
        ...     warmup_steps=100,
        ...     reset_every=1000,
        ...     base_scheduler=base_scheduler
        ... )
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        reset_every,
        base_scheduler=None,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.reset_every = reset_every
        self.base_scheduler = base_scheduler
        self.last_reset = -1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        steps_since_reset = self.last_epoch - self.last_reset

        # Check if we need to reset
        if (
            self.reset_every
            and self.last_epoch > 0
            and self.last_epoch % self.reset_every == 0
        ):
            self.last_reset = self.last_epoch
            steps_since_reset = 0

        # Get target learning rates from base scheduler
        if self.base_scheduler is None:
            target_lrs = self.base_lrs
        else:
            target_lrs = self.base_scheduler.get_lr()

        # Apply warmup if we're in a warmup period
        if steps_since_reset < self.warmup_steps:
            # Linear warmup from 0 to target_lr
            warmup_factor = steps_since_reset / self.warmup_steps
            return [lr * warmup_factor for lr in target_lrs]

        return target_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            learning_rates = self.get_lr()
            if self.base_scheduler is not None:
                self.base_scheduler.step(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, learning_rates):
            param_group["lr"] = lr
