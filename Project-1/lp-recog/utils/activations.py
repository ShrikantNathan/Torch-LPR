import torch
import torch.nn as nn
import torch.nn.functional as Fn
from typing import Any


class SigmoidLinearUnit(nn.Module):
    @staticmethod
    def forward(x):
        return torch.multiply(x, torch.sigmoid(x))


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return torch.multiply(x, torch.divide(Fn.hardtanh(x + 3, 0., 6.), 6))


class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any):
            ctx.save_for_backward(*args)
            return torch.multiply(*args, torch.sigmoid(*args))

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return torch.multiply(*grad_outputs, torch.multiply(sx, torch.add(1, torch.multiply(x, (1 - sx)))))

    def forward(self, x):
        self.F.apply(x)


class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return torch.multiply(x, Fn.softplus(x).tanh())


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(Fn.softplus(x)))

        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = Fn.softplus(*grad_outputs).tanh()
            return torch.multiply(*grad_outputs, torch.multiply(torch.add(
                fx, torch.multiply(x, sx)), torch.multiply((1 - fx), fx)))

    def forward(self, x):
        return self.F.apply(x)


class FReLU(nn.Module):
    def __init__(self, c1, k=(3, 3)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=k,
                              stride=(1, 1), padding=1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))