import torch as th
import torch.nn as nn

from lib.loss import Loss as _Loss, LossTerm, UCO, intermediate_kernels


class ReconstructionLoss(LossTerm):
    def __call__(self, net, cfg, extra):
        return nn.functional.mse_loss(input=net.reconstruction.view(net.input.size()), target=net.input)


class KLLoss(LossTerm):
    def __init__(self, cfg):
        super().__init__()
        self.loss_func = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def __call__(self, net, cfg, extra):
        assert net.current_batch_idx is not None, "Got net.current_batch_idx == None. Net has to be called with idx " \
                                                  "in order to evaluate KL-loss."
        target = net.target_dist[net.current_batch_idx]
        loss = self.loss_func(th.log(net.output), target)
        return loss


class Loss(_Loss):
    TERM_CLASSES = {
        "kl": KLLoss,
        "reconstruct": ReconstructionLoss,
        "uco": UCO,
    }
    EXTRA_FUNCS = {
        "intermediate_kernels": intermediate_kernels
    }
