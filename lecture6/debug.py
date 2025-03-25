import torch

class DebugOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        return x

    @staticmethod
    def symbolic(g, x, name):
        return g.op('my::Debug', x, name_s=name)

debug_apply = DebugOp.apply