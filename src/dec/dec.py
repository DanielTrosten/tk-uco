import numpy as np
import torch as th
import torch.nn as nn
from sklearn.cluster import KMeans

import helpers
from models.model_base import ModelBase
from lib.backbones import create_backbone
from lib.kernel import cdist
from lib.objective_function_mismatch import calc_ofm
from lib.metrics import ordered_cmat
from dec.loss import Loss



class ClusteringModule(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()

        self.cfg = cfg
        self.centroids = nn.Parameter(th.normal(mean=th.zeros(cfg.n_clusters, input_size[0]), std=1),
                                      requires_grad=True)
        self.pow = -1 * (self.cfg.alpha + 1) / 2

    def init_centroids(self, data, labels=None):
        data = th.flatten(data, start_dim=1)
        kmeans = KMeans(n_clusters=self.cfg.n_clusters)
        predictions = kmeans.fit_predict(helpers.npy(data))

        if labels is not None:
            # Only use labels to evaluate k-means accuracy.
            acc, cmat = ordered_cmat(helpers.npy(labels), predictions)
            print("k-means accuracy:", acc)
            print("k-means confusion matrix:\n", cmat)

        self.centroids.data = th.tensor(kmeans.cluster_centers_.astype(np.float32)).type_as(self.centroids.data)

    def forward(self, z):
        z = th.flatten(z, start_dim=1)
        dists = cdist(z, self.centroids)
        sim = (1 + dists / self.cfg.alpha) ** self.pow
        norm_fac = sim.sum(dim=1, keepdim=True)
        sim = sim / norm_fac
        return sim


class DEC(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.is_pre_train = True
        self.enable_automatic_optimization = False
        self.reconstruct_in_fine_tune = "reconstruct" in self.cfg.fine_tune_loss_config.funcs

        if cfg.dropout_prob > 0:
            self.dropout = nn.Dropout(p=cfg.dropout_prob, inplace=False)
        else:
            self.dropout = nn.Identity()

        self.backbone = create_backbone(cfg.backbone_config, flatten_output=False)
        self.clustering_module = ClusteringModule(cfg.cm_config, input_size=[np.prod(self.backbone.output_size)])
        self.decoder = create_backbone(cfg.decoder_config, input_size=self.backbone.output_size, flatten_output=False)

        self.pre_train_loss = Loss(cfg.pre_train_loss_config)
        self.fine_tune_loss = Loss(cfg.fine_tune_loss_config)

        self.input = self.current_batch_idx = self.noisy_input = self.hidden = self.output = self.reconstruction = None

        # Initialize weights.
        # self.apply(helpers.he_init_weights)

    def configure_optimizers(self):
        pre_train_params = list(self.backbone.parameters()) + list(self.decoder.parameters())
        pre_train_optimizer = self._optimizer_from_cfg(self.cfg.pre_train_optimizer_config, pre_train_params)

        fine_tune_params = list(self.backbone.parameters()) + list(self.clustering_module.parameters())
        if self.reconstruct_in_fine_tune:
            fine_tune_params += list(self.decoder.parameters())
        fine_tune_optimizer = self._optimizer_from_cfg(self.cfg.fine_tune_optimizer_config, fine_tune_params)
        return pre_train_optimizer, fine_tune_optimizer

    @staticmethod
    def _print_msg(msg, sep=(100 * "=")):
        print(f"{sep}\n{msg}\n{sep}")

    def _init_fine_tune(self):
        self._print_msg("Pre-training done. Initializing centroids.")

        hidden, labels = [], []
        for data, label, _ in self.train_loader:
            _ = self(data.to(self.device))
            hidden.append(self.hidden.detach())
            labels.append(label)
        hidden = th.cat(hidden, dim=0)
        labels = th.cat(labels, dim=0)

        self.clustering_module.init_centroids(hidden, labels)
        self.is_pre_train = False
        self.reconstruction = None

    @staticmethod
    def _target_dist(q):
        f = q.sum(dim=0, keepdim=True)
        un_normed = (q ** 2) / f
        normed = un_normed / un_normed.sum(dim=1, keepdim=True)
        return normed

    def _update_target_dist(self):
        self._print_msg("Updating target distribution")
        inds, predictions = [], []
        for data, _, idx in self.train_loader:
            inds.append(idx)
            predictions.append(self(data.to(self.device)).detach())

        predictions = th.cat(predictions, dim=0)[th.argsort(th.cat(inds, dim=0))]
        self.target_dist = self._target_dist(predictions)

    def _pre_train_forward(self, x):
        self.input = x
        # print("input", np.isnan(helpers.npy(self.input)).any())

        self.hidden = self.backbone(self.dropout(self.input))
        # print("hidden", np.isnan(helpers.npy(self.hidden)).any())

        self.reconstruction = self.decoder(self.hidden)
        # print("reconstruction", np.isnan(helpers.npy(self.reconstruction)).any())

        self.output = self.clustering_module(self.hidden)
        # print("output", np.isnan(helpers.npy(self.output)).any())

        return self.output

    def _fine_tune_forward(self, x):
        self.input = x
        self.hidden = self.backbone(self.input)
        self.output = self.clustering_module(self.hidden)

        if self.reconstruct_in_fine_tune:
            self.reconstruction = self.decoder(self.hidden)

        return self.output

    def forward(self, x, idx=None):
        self.current_batch_idx = idx
        if self.is_pre_train:
            return self._pre_train_forward(x)
        return self._fine_tune_forward(x)

    def get_loss(self):
        if self.is_pre_train:
            return self.pre_train_loss(self)
        return self.fine_tune_loss(self)

    def get_ofm(self, data):
        if self.is_pre_train:
            return np.nan
        return calc_ofm(data, self)

    def training_step(self, batch, idx, optimizer_idx):
        _ = self(batch[0], batch[2])

        opt = self.optimizers()[0] if self.is_pre_train else self.optimizers()[1]
        losses = self.get_loss()

        self._log_dict({"epoch": self.current_epoch}, prefix="")
        self._log_dict(losses, prefix="train_loss")

        self.manual_backward(losses["tot"], opt)
        opt.step()
        # return losses["tot"]

    def training_epoch_end(self, outputs):
        if (not self.is_pre_train) and ((self.current_epoch - self.cfg.n_pre_train_epochs) %
                                        self.cfg.target_dist_update_interval) == 0:
            # Update the target distribution
            self._update_target_dist()

        if self.current_epoch == self.cfg.n_pre_train_epochs:
            # Initialize the fine-tuning stage
            self._init_fine_tune()
            self._update_target_dist()

        # print(self.clustering_module.centroids)
