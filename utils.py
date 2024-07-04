from typing import Union
import torch
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

def get_trainer(
        dataset_name_or_id: Union[int, str],
        configuration: str,
        fold: int,
        # trainer_name: str = 'nnUNetTrainer',
        plans_identifier: str = 'nnUNetPlans',
        use_compressed: bool = False,
        device: torch.device = torch.device('cuda'),
        # *args,
        # **kwargs
):
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    # import src.trainers as trainers
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer_class = nnUNetTrainer # getattr(trainers, trainer_name)
    nnunet_trainer = nnunet_trainer_class(
        plans=plans, configuration=configuration, fold=fold,
        dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device,
        # *args, **kwargs
    )
    return nnunet_trainer

def uncertainty(p_hat, var='sum'):
    p_mean = torch.mean(p_hat, dim=0)
    ale = torch.mean(p_hat*(1-p_hat), dim=0)
    epi = torch.mean(p_hat**2, dim=0) - p_mean**2
    if var == 'sum':
        ale = torch.sum(ale, dim=0)
        epi = torch.sum(epi, dim=0)
    elif var == 'top':
        ale = ale[torch.argmax(p_mean)]
        epi = epi[torch.argmax(p_mean)]
    uncert = ale + epi
    return p_mean, uncert, ale, epi


def hard_dice_score(pred, target):
    tp, fp, fn, _ = get_tp_fp_fn_tn(pred, target, axes=(0,2,3))
    tp = tp.detach().cpu().numpy()
    fp = fp.detach().cpu().numpy()
    fn = fn.detach().cpu().numpy()
    dice = 2 * tp / (2 * tp + fp + fn + 1)
    return dice

def onehot(t, n_cls):
    b, _, h, w = t.shape
    t_onehot = torch.zeros((b, n_cls, h, w)).to(t.device)
    t_onehot.scatter_(1, t.long(), 1)
    return t_onehot

def onehot3d(t, n_cls):
    b, _, d, h, w = t.shape
    t_onehot = torch.zeros((b, n_cls, d, h, w)).to(t.device)
    t_onehot.scatter_(1, t.long(), 1)
    return t_onehot

def hard_dice_score3d(pred, target):
    tp, fp, fn, _ = get_tp_fp_fn_tn(pred, target, axes=(0,2,3,4))
    tp = tp.detach().cpu().numpy()
    fp = fp.detach().cpu().numpy()
    fn = fn.detach().cpu().numpy()
    dice = 2 * tp / (2 * tp + fp + fn + 1)
    return dice

def compute_dice_score(pred, target, n_cls):
    pred_onehot = onehot(pred, n_cls=n_cls)
    target_onehot = onehot(target, n_cls=n_cls)
    # We might miss the fact that the classifier does not predict a label though it was trained on it
    # present_labels = present_target_labels * present_pred_labels
    dice = hard_dice_score(pred_onehot, target_onehot)
    return dice

def _sum(tensor_list):
    tensor_sum = torch.zeros_like(tensor_list[0])
    for t in tensor_list:
        tensor_sum += t
    return tensor_sum

def merged_logits_from_sep_cls(logits):
    n_preds_per_label = torch.stack([l.sum((0,2,3)) > 0 for l in logits]).sum(0)
    merged_logits = (1 / n_preds_per_label).view(-1,1,1) * _sum(logits)
    return merged_logits

def merged_logits_from_sep_cls_with_uncert(logits, uncert):
    merged_logits = merged_logits_from_sep_cls(logits)
    merged_logits[:,0] *= (1 - _sum(uncert))
    return merged_logits