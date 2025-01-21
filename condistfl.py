# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from typing import List, Dict
from copy import deepcopy
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class ConDistTransform(object):
    def __init__(
        self,
        num_classes: int,
        foreground: Sequence[int],
        background: Sequence[Union[int, Sequence[int]]],
        temperature: float = 2.0,
    ):
        self.num_classes = num_classes

        self.foreground = foreground
        self.background = background

        if temperature < 0.0:
            raise ValueError("Softmax temperature must be a postive number!")
        self.temperature = temperature

    def softmax(self, data: Tensor):
        return torch.softmax(data / self.temperature, dim=1)

    def reduce_channels(self, data: Tensor, eps: float = 1e-5):
        batch, channels, *shape = data.shape
        if channels != self.num_classes:
            raise ValueError(f"Expect input with {self.num_classes} channels, get {channels}")

        fg_shape = [batch] + [1] + shape
        bg_shape = [batch] + [len(self.background)] + shape

        # Compute the probability for the union of local foreground
        fg = torch.zeros(fg_shape, dtype=torch.float32, device=data.device)
        for c in self.foreground:
            fg += data[:, c, ::].view(*fg_shape)

        # Compute the raw probabilities for each background group
        bg = torch.zeros(bg_shape, dtype=torch.float32, device=data.device)
        for i, g in enumerate(self.background):
            if isinstance(g, int):
                bg[:, i, ::] = data[:, g, ::]
            else:
                for c in g:
                    bg[:, i, ::] += data[:, c, ::]

        # Compute condistional probability for background groups
        return bg / (1.0 - fg + eps)

    def generate_mask(self, targets: Tensor, ground_truth: Tensor):
        targets = torch.argmax(targets, dim=1, keepdim=True)

        # The mask covers the background but excludes false positive areas
        condition = torch.zeros_like(targets, device=targets.device)
        for c in self.foreground:
            condition = torch.where(torch.logical_or(targets == c, ground_truth == c), 1, condition)
        mask = 1 - condition

        return mask.float() # astype(torch.float32)

    def __call__(self, preds: Tensor, targets: Tensor, ground_truth: Tensor) -> Tuple[Tensor]:
        mask = self.generate_mask(targets, ground_truth)

        preds = self.softmax(preds)
        preds = self.reduce_channels(preds)

        targets = self.softmax(targets)
        targets = self.reduce_channels(targets)

        return preds, targets, mask


class MarginalTransform(object):
    def __init__(self, foreground: Sequence[int], softmax: bool = False):
        self.foreground = foreground
        self.softmax = softmax

    def reduce_background_channels(self, tensor: Tensor) -> Tensor:
        n_chs = tensor.shape[1]
        slices = torch.split(tensor, 1, dim=1)

        fg = [slices[i] for i in self.foreground]
        bg = sum([slices[i] for i in range(n_chs) if i not in self.foreground])

        output = torch.cat([bg] + fg, dim=1)
        return output

    def __call__(self, preds: Tensor, target: Tensor) -> Tuple[Tensor]:
        n_pred_ch = preds.shape[1]
        if n_pred_ch == 1:
            # Marginal loss is not intended for single channel output
            return preds, target

        if self.softmax:
            preds = torch.softmax(preds, 1)

        if target.shape[1] == 1:
            target = one_hot(target, num_classes=n_pred_ch)
        elif target.shape[1] != n_pred_ch:
            raise ValueError(f"Number of channels of label must be 1 or {n_pred_ch}.")

        preds = self.reduce_background_channels(preds)
        target = self.reduce_background_channels(target)

        return preds, target


class ConDistDiceLoss(_Loss):
    def __init__(
        self,
        num_classes: int,
        foreground: Sequence[int],
        background: Sequence[Union[int, Sequence[int]]],
        temperature: float = 2.0,
        include_background: bool = True,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__()

        self.transform = ConDistTransform(num_classes, foreground, background, temperature=temperature)
        self.dice = MaskedDiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

    def forward(self, preds: Tensor, targets: Tensor, ground_truth: Tensor):
        n_chs = preds.shape[1]
        if (ground_truth.shape[1] > 1) and (ground_truth.shape[1] == n_chs):
            ground_truth = torch.argmax(ground_truth, dim=1, keepdim=True)
        preds, targets, mask = self.transform(preds, targets, ground_truth)
        return self.dice(preds, targets, mask=mask)


class MarginalDiceCELoss(_Loss):
    def __init__(
        self,
        foreground: Sequence[int],
        include_background: bool = True,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ):
        super().__init__()

        self.transform = MarginalTransform(foreground, softmax=softmax)
        self.dice_ce = DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            ce_weight=ce_weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(self, preds: Tensor, targets: Tensor):
        preds, targets = self.transform(preds, targets)
        return self.dice_ce(preds, targets)


class MoonContrasiveLoss(torch.nn.Module):
    def __init__(self, tau: float = 1.0):
        super().__init__()

        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def forward(self, z: Tensor, z_prev: Tensor, z_glob: Tensor):
        sim_prev = F.cosine_similarity(z, z_prev, dim=1)
        sim_glob = F.cosine_similarity(z, z_glob, dim=1)

        exp_prev = torch.exp(sim_prev / self.tau)
        exp_glob = torch.exp(sim_glob / self.tau)

        loss = -torch.log(exp_glob / (exp_glob + exp_prev))
        return loss.mean()


class ConDistFLLabelTransform:
    def __init__(self, ds_labels: Dict[str, int], all_labels: Dict[str, int]):
        self.ds_labels = ds_labels
        self.all_labels = all_labels

    def __call__(self, data, target, *args, **kwargs):
        # assume one channel seg map
        try:
            b = data.size(0)
            if type(target) == list:
                seg_new = [torch.zeros(len(self.all_labels)+1, b, *l.shape[2:]) for l in target]
                # seg_new = [torch.zeros_like(l) for l in target]
            else:
                seg_new = [torch.zeros(len(self.all_labels)+1, b, *target.shape[2:])]
                # seg_new = [torch.zeros_like(target)]
            for t, t_new in zip(target, seg_new):
                t_new[0] = (t[:,0] == 0).float()
                for l, i in self.all_labels.items():
                    if l in self.ds_labels:
                        j = int(self.ds_labels[l])
                        t_new[i][t[:,0]==j] = 1.
                        # t_new[t==j] = i

            seg_new = [t.permute(1,0,*[i for i in range(2,len(data.shape))]) for t in seg_new]
        except:
            import pdb;pdb.set_trace()
        return {'data': data, 'target': seg_new}


class ConDistFLTrainer(nnUNetTrainer):
    def __init__(
            self, 
            foreground: List[int],
            background: List[int],
            weight_range: List[int] = [0.01, 1.00],
            max_rounds: int = 1000,
            temperature: float = 0.5,
            num_classes: int = 12,
            fedprox_mu: Optional[float] = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            # foreground=foreground, 
            # background=background,
            # weight_range=weight_range,
            # max_rounds=max_rounds,
            # temperature=temperature,
            # num_classes=num_classes,
            *args, **kwargs
        )
        self.foreground = foreground
        self.background = background
        self.weight_range = weight_range
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.num_classes = num_classes
        self.fedprox_mu = fedprox_mu
    # sch = CosineAnnealingLR(self.opt, T_max=self.max_steps, eta_min=1e-7)

    def _build_loss(self):
        from monai.losses import DeepSupervisionLoss
        condist_loss_fn = ConDistDiceLoss(
            self.num_classes, self.foreground, self.background, temperature=self.temperature, smooth_nr=0.0, batch=True
        )
        marginal_loss_fn = MarginalDiceCELoss(self.foreground, softmax=True, smooth_nr=0.0, batch=True)
        ds_loss_fn = DeepSupervisionLoss(marginal_loss_fn, weights=[0.5333, 0.2667, 0.1333, 0.0667])
        loss = {'ds_loss_fn': ds_loss_fn, 'condist_loss_fn': condist_loss_fn}
        return loss
    
    def update_condist_weight(self):
        left = min(self.weight_range)
        right = max(self.weight_range)
        intv = (right - left) / (self.max_rounds - 1)
        self.weight = left + intv * self.current_epoch # round
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.global_model = deepcopy(self.network)
        self.grad_org = self.global_model.state_dict()
        self.update_condist_weight()
    
    def on_epoch_end(self):
        super().on_epoch_end()
        del self.global_model
        del self.grad_org
    
    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            ds_loss = self.loss['ds_loss_fn'](output, target[0])
            
            with torch.no_grad():
                teacher_target = self.global_model(data)
            if type(teacher_target) == list:
                teacher_target = teacher_target[0]

            # no DS here?
            condist_loss = self.loss['condist_loss_fn'](output[0], teacher_target, target[0])

            l = ds_loss + self.weight * condist_loss
            
            if self.fedprox_mu is not None:
                reg_loss = 0.0
                for name, param in self.network.named_parameters():
                    if 'weight' in name and name in self.grad_org:
                        reg_loss += torch.norm(param-self.grad_org[name], 2)    
                reg_loss = reg_loss*0.5*self.fedprox_mu
                l += reg_loss
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            # l = self.loss(output, target)
            ds_loss = self.loss['ds_loss_fn'](output, target[0])

            with torch.no_grad():
                teacher_target = self.global_model(data)
            if type(teacher_target) == list:
                teacher_target = teacher_target[0]

            # no DS here?
            condist_loss = self.loss['condist_loss_fn'](output[0], teacher_target, target[0])

            l = ds_loss + self.weight * condist_loss
        
        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))
        
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg
        
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}