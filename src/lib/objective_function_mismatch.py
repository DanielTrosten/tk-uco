import torch as th


def get_cluster_and_companion_loss(losses):
    l_cluster = losses["tot"].new_tensor(0.0)
    l_companion = losses["tot"].new_tensor(0.0)

    for loss_name, loss_value in losses.items():
        if loss_name.startswith("uco") or loss_name.startswith("reconstruct"):
            l_companion += loss_value
        elif loss_name != "tot":
            l_cluster += loss_value

    return {"cluster": l_cluster, "companion": l_companion}


def _flat_cat(tup):
    return th.cat([th.flatten(elem) for elem in tup], dim=0)


@th.enable_grad()
def get_grad(batch, net, loss_type):
    assert loss_type in ("companion", "cluster")
    _ = net(batch[0], idx=batch[2])
    losses = net.get_loss()
    loss = get_cluster_and_companion_loss(losses)[loss_type]
    params = net.backbone.parameters()
    grad = th.autograd.grad(outputs=loss, inputs=params)
    return _flat_cat(grad)


def calc_ofm(batch, net):
    grad_cluster = get_grad(batch, net, "cluster")
    grad_companion = get_grad(batch, net, "companion")
    grad_cluster = th.nn.functional.normalize(grad_cluster, dim=0)
    grad_companion = th.nn.functional.normalize(grad_companion, dim=0)
    ofm = grad_cluster @ grad_companion
    return ofm


