from typing import Tuple, List
from pathlib import Path
import shutil
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import SimpleITK as sitk
import pandas as pd
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


def copy_dataset_jsons(src: Path, dst: Path):
    datasets_in_dst = [x.name for x in dst.iterdir()]
    for dataset_json_path in src.glob('*/dataset.json'):
        dataset_name = dataset_json_path.parent.name
        if dataset_name not in datasets_in_dst:
            continue
        shutil.copy(dataset_json_path, dst / dataset_name)


def create_batches(
        dataset_id: int,
        dst: Path,
        configuration: str = '2d',
        fold: int = 1,
        patch_size: Tuple[int] = (320,320)
):
    from utils import get_trainer
    trainer = get_trainer(dataset_id, configuration, fold)

    dst /= trainer.plans_manager.dataset_name
    dst.mkdir(exist_ok=True)
    dst /= f'validation_fold{1}'
    # dst /= f'training_fold{1}'
    dst.mkdir(exist_ok=True)
    (dst / 'images').mkdir(exist_ok=True)
    (dst / 'labels').mkdir(exist_ok=True)
    # trainer.configuration_manager.configuration['patch_size'] = patch_size
    dataloader_train, dataloader_val = trainer.get_dataloaders()

    # only transferred to torch and downsampled for loss computation -> not needed
    # val_transform = dataloader_val.transform
    val_loader = dataloader_val.data_loader
    # val_loader = dataloader_train.data_loader
    val_loader.patch_size = patch_size
    indices = val_loader.indices
    for i, case in enumerate(tqdm(indices)):

        data, seg, properties = val_loader._data.load_case(case)
        data_all = np.zeros((1, data.shape[1], *patch_size))
        seg_all = np.zeros((1, seg.shape[1], *patch_size))
        for slice_idx in range(data.shape[1]):
            data_slice, seg_slice = data[:,slice_idx], seg[:,slice_idx]

            force_fg = False
            class_locations = None
            selected_class_or_region = None
            shape = data_slice.shape[1:]

            dim = len(shape)
            bbox_lbs, bbox_ubs = val_loader.get_bbox(
                shape, force_fg if selected_class_or_region is not None else None,
                class_locations, overwrite_class=selected_class_or_region)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data_slice.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data_slice = data_slice[this_slice]

            this_slice = tuple([slice(0, seg_slice.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg_slice = seg_slice[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_slice = np.pad(data_slice, ((0, 0), *padding), 'constant', constant_values=0)
            data_all[:,slice_idx] = data_slice
            # changed -1 to 0 -> only needed in cascaded nnUNet pipeline
            seg_slice = np.pad(seg_slice, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[:,slice_idx] = seg_slice
        
        data_sitk = sitk.GetImageFromArray(data_all[0])
        seg_sitk = sitk.GetImageFromArray(seg_all[0])
        data = torch.from_numpy(data_all).float()
        seg = torch.from_numpy(seg_all).float()

        torch.save(data, dst / 'images' / f'{case}.pt')
        torch.save(seg, dst / 'labels' / f'{case}.pt')

        sitk.WriteImage(data_sitk, dst / 'images' / f'{case}.nii.gz')
        sitk.WriteImage(seg_sitk, dst / 'labels' / f'{case}.nii.gz')


class Model(nn.Module):
    def __init__(self, batch_size: int, trainer_name: str, dataset_names: List[str], LABEL_SUBSETS):
        super().__init__()
        self.batch_size = batch_size

        from train import get_trainer
        master_trainer = get_trainer(
            dataset_name_or_id='Dataset100_CT20K',
            configuration='2d',
            fold=1,
            trainer_name=trainer_name
        )
        dropout_p = 0.3
        configuration = '2d'
        if dropout_p is not None:
            master_trainer.configuration_manager.network_arch_init_kwargs['dropout_op'] = f'torch.nn.Dropout{configuration[:2]}'
            master_trainer.configuration_manager.network_arch_init_kwargs['dropout_op_kwargs'] = {'p': dropout_p}
        master_trainer.initialize()
        self.backbone = master_trainer.network.cpu()
        self.cls_heads = nn.ModuleDict({})
        for dataset_name in dataset_names:
            n_labels = len(LABEL_SUBSETS[dataset_name])
            seg_head = nn.ModuleList([nn.Conv2d(m.in_channels, n_labels+1, m.kernel_size, m.stride) for m in self.backbone.decoder.seg_layers])
            self.cls_heads[dataset_name] = seg_head
        
        self.backbone.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in self.backbone.decoder.seg_layers])
        # self.cls_heads = nn.ModuleDict({
        #     dataset_name: nn.Conv2d(ch, n_labels, kernel_size=(1,1), stride=(1,1)) for dataset_name in dataset_names
        # })
    
    def forward(self, x):
        out = self.backbone(x)
        outs = {}
        for n, cls_head in self.cls_heads.items():
            outs[n] = cls_head[-1](out)
        # outs = {n: cls_head[:-1](out) for i, (n, cls_head) in enumerate(self.cls_heads.items())}
        return outs


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

def merged_logits_from_sep_cls_with_uncert(logits, uncert, alpha=1.):
    merged_logits = merged_logits_from_sep_cls(logits)
    merged_logits[:,0] *= (1 - _sum(uncert))
    return merged_logits


def test(
        exp_name,
        LABEL_SUBSETS,
        model,
        train_dataset_ids: List[int],
        test_dataset_ids: List[int],
        src: Path,
        with_FUN: bool = True
):

    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    test_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in test_dataset_ids]

    train_labels = [LABEL_SUBSETS[ds_name] for ds_name in train_dataset_names]
    train_labels = np.unique([xx for x in train_labels for xx in x])
    train_labels = {l: i+1 for i, l in enumerate(train_labels)}
    n_cls = len(train_labels)+1

    test_dataset_labels = {}
    for ds_name in test_dataset_names:
        with open(src / ds_name / 'dataset.json', 'r') as f:
            test_dataset_labels[ds_name] = json.load(f)['labels']
        
        if ds_name == 'Dataset014_learn2reg':
            del test_dataset_labels[ds_name]['urinary bladder']
            test_dataset_labels[ds_name]['spleen'] = '2'

    Path(exp_name).mkdir(exist_ok=True)

    mc_iter = 10
    dices = []
    for test_ds_name in tqdm(test_dataset_names):
        ds_labels = test_dataset_labels[test_ds_name]
        
        intersection_labels = [t for t in train_labels if t in ds_labels]
        if not len(intersection_labels):
            continue
        print(test_ds_name)
        _fold = 'validation_fold1'
        img_paths = list((src / test_ds_name / _fold / 'images').glob('*.pt'))
        img_paths = sorted(img_paths)
       
        for i, img_path in enumerate(tqdm(img_paths, leave=False)):
            if i == 20:
                break

            img = torch.load(img_path)
            seg_path = src / test_ds_name / _fold / 'labels' / img_path.name
            seg = torch.load(seg_path)
            case = img_path.stem

            seg_new = torch.zeros_like(seg)
            for l, i in train_labels.items():
                if l in ds_labels:
                    j = int(ds_labels[l])
                    seg_new[seg == j] = i
            
            merged_logits_all, merged_logits_fun_all, uncerts_all, inputs_all, targets_all = [], [], [], [], []
            ds_logits = {}
            preds = {}

            idxs = seg_new.sum((0,2,3)) > 0
            img_new_slices, seg_new_slices = img.permute(1,0,2,3)[idxs], seg_new.permute(1,0,2,3)[idxs]

            for img_slice, seg_slice in zip(tqdm(img_new_slices[::5], leave=False), seg_new_slices[::5]):
                inputs_all.append(img_slice)
                targets_all.append(seg_slice)
                with torch.no_grad():
                    mc_outputs = [{k: torch.softmax(v.cpu(), dim=1) for k, v in model(img_slice.cuda()[None]).items()} for _ in range(mc_iter)]
                # collate
                mc_outputs_collated = {}
                for train_ds_name in train_dataset_names:
                    mc_outputs_collated[train_ds_name] = torch.cat([o[train_ds_name] for o in mc_outputs], dim=0)
                
                mean_logits, uncerts = [], []
                for train_ds_name, mc_outputs in mc_outputs_collated.items():
                    mean, uncert, ale, epi = uncertainty(mc_outputs, var='sum')

                    mean_new = torch.zeros(len(train_labels)+1, mean.size(1), mean.size(2))
                    mean_new[0] = mean[0]
                    ds_label_subset = np.array(LABEL_SUBSETS[train_ds_name])

                    if train_ds_name == 'Dataset014_learn2reg':
                        ds_label_subset[-1] = 'spleen'

                    for k, i in train_labels.items():
                        if k in ds_label_subset:
                            mean_new[i] = mean[np.argwhere(ds_label_subset == k)[0,0]+1]
                    mean_logits.append(mean_new[None])
                    uncerts.append(uncert)

                    if train_ds_name not in preds:
                        preds[train_ds_name] = []
                    preds[train_ds_name].append(mean_new.argmax(0, keepdim=True))

                    if train_ds_name not in ds_logits:
                        ds_logits[train_ds_name] = []
                    ds_logits[train_ds_name].append(mean_new[None])
                
                if len(train_dataset_ids) > 1:
                    merged_logits = merged_logits_from_sep_cls(mean_logits)
                    merged_logits_fun = merged_logits_from_sep_cls_with_uncert(mean_logits, uncerts)

                    merged_logits_all.append(merged_logits)
                    merged_logits_fun_all.append(merged_logits_fun)
                    uncerts_all.append(_sum(uncerts))

                    merged_pred = merged_logits.argmax(dim=1, keepdim=True)
                    merged_pred_fun = merged_logits_fun.argmax(dim=1, keepdim=True)

                    target_onehot = onehot(seg_slice[None], n_cls=n_cls)
                    merged_pred_onehot = onehot(merged_pred, n_cls=n_cls)
                    merged_pred_fun_onehot = onehot(merged_pred_fun, n_cls=n_cls)

                    dice = hard_dice_score(merged_pred_onehot, target_onehot)[1:]
                    dice_fun = hard_dice_score(merged_pred_fun_onehot, target_onehot)[1:]

                    mask = np.zeros((12,))
                    mask[(torch.unique(seg_slice)[1:]-1).long()] = 1
                    mask = mask.astype(bool)
                    # then ignored py pandas
                    dice[~mask] = np.nan
                    dice_fun[~mask] = np.nan
                    print(case)
                    print(np.nanmean(dice[mask]))
                    print(np.nanmean(dice_fun[mask]))

                    dice = {l: d for l, d in zip(train_labels, dice)}
                    dice_fun = {f'{l}_fun': d for l, d in zip(train_labels, dice_fun)}

                    dices.append({'case': f'{case}_{i}', 'test_dataset': test_ds_name, **dice, **dice_fun})

                else:
                    merged_logits_all.append(mean_logits[0])
                    uncerts_all.append(uncerts[0])
                    pred = mean_logits[0].argmax(dim=1, keepdim=True)
                    uncert = uncerts[0]

                    target_onehot = onehot(seg_slice[None], n_cls=n_cls)
                    pred_onehot = onehot(pred, n_cls=n_cls)

                    dice = hard_dice_score(pred_onehot, target_onehot)[1:]
                    dice = {l: d for l, d in zip(train_labels, dice)}

                    dices.append({'case': f'{case}_{i}', 'test_dataset': test_ds_name, **dice})
            
            if len(merged_logits_all) == 0:
                continue

            if len(train_dataset_ids) > 1:
                merged_logits_all = torch.cat(merged_logits_all, dim=0).permute(1,0,2,3)[None].numpy()
                merged_logits_fun_all = torch.cat(merged_logits_fun_all, dim=0).permute(1,0,2,3)[None].numpy()
                uncerts_all = torch.stack(uncerts_all).numpy()
                inputs_all = torch.cat(inputs_all, dim=0).numpy()
                targets_all = torch.cat(targets_all, dim=0).numpy()

                torch.save(merged_logits_all, Path(exp_name) / f'{test_ds_name}_{case}_merged_logits.pt')
                torch.save(merged_logits_fun_all, Path(exp_name) / f'{test_ds_name}_{case}_merged_logits_fun.pt')

                for ds_name, ds_pred in preds.items():
                    ds_pred = torch.cat(ds_pred, dim=0)
                    ds_pred = sitk.GetImageFromArray(ds_pred.numpy().astype(np.uint8))
                    sitk.WriteImage(ds_pred, Path(exp_name) / f'{test_ds_name}_{case}_{ds_name}_pred.nii.gz')

                ml = sitk.GetImageFromArray(merged_logits_all.argmax(axis=1).astype(np.uint8)[0])
                ml_fun = sitk.GetImageFromArray(merged_logits_fun_all.argmax(axis=1).astype(np.uint8)[0])
                u = sitk.GetImageFromArray(uncerts_all.astype(np.float32))
                inputs = sitk.GetImageFromArray(inputs_all.astype(np.float32))
                targets = sitk.GetImageFromArray(targets_all.astype(np.uint8))
                sitk.WriteImage(ml, Path(exp_name) / f'{test_ds_name}_{case}_ml.nii.gz')
                sitk.WriteImage(ml_fun, Path(exp_name) / f'{test_ds_name}_{case}_ml_fun.nii.gz')
                sitk.WriteImage(u, Path(exp_name) / f'{test_ds_name}_{case}_u.nii.gz')
                sitk.WriteImage(inputs, Path(exp_name) / f'{test_ds_name}_{case}_inputs.nii.gz')
                sitk.WriteImage(targets, Path(exp_name) / f'{test_ds_name}_{case}_targets.nii.gz')

                for ds_name, logits in ds_logits.items():
                    torch.save(logits, Path(exp_name) / f'{test_ds_name}_{case}_{ds_name}_logits.pt')
            else:
                merged_logits_all = torch.cat(merged_logits_all, dim=0).permute(1,0,2,3)[None].numpy()
                # merged_logits_fun_all = torch.cat(merged_logits_fun_all, dim=0).permute(1,0,2,3)[None].numpy()
                uncerts_all = torch.stack(uncerts_all).numpy()
                inputs_all = torch.cat(inputs_all, dim=0).numpy()
                targets_all = torch.cat(targets_all, dim=0).numpy()

                ml = sitk.GetImageFromArray(merged_logits_all.argmax(axis=1).astype(np.uint8)[0])
                # ml_fun = sitk.GetImageFromArray(merged_logits_fun_all.argmax(axis=1).astype(np.uint8)[0])
                u = sitk.GetImageFromArray(uncerts_all.astype(np.float32))
                inputs = sitk.GetImageFromArray(inputs_all.astype(np.float32))
                targets = sitk.GetImageFromArray(targets_all.astype(np.uint8))
                sitk.WriteImage(ml, Path(exp_name) / f'{test_ds_name}_{case}_ml.nii.gz')
                # sitk.WriteImage(ml_fun, Path(exp_name) / f'{test_ds_name}_{case}_ml_fun.nii.gz')
                sitk.WriteImage(u, Path(exp_name) / f'{test_ds_name}_{case}_u.nii.gz')
                sitk.WriteImage(inputs, Path(exp_name) / f'{test_ds_name}_{case}_inputs.nii.gz')
                sitk.WriteImage(targets, Path(exp_name) / f'{test_ds_name}_{case}_targets.nii.gz')

    
            df_dice = pd.DataFrame(dices)
            df_dice.to_csv(f'{exp_name}_dice.csv')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['nnUNet_n_proc_DA'] = '0'
    os.environ['nnUNet_raw'] = '../../data/CT20K/ct_ready_nnunet/raw'
    os.environ['nnUNet_preprocessed'] = '../../data/CT20K/ct_ready_nnunet/preprocessed_global2'
    os.environ['nnUNet_results'] = './tmp'

    src = Path('/mnt/sds/sd20i001/malte/data/CT20K/nnUNet_preprocessed_batches2')
    # src.mkdir(exist_ok=True)

    # for dataset_id in [1,2,4,5,6,7,9,14,37,38]:# 1,2,
    #     create_batches(
    #         dataset_id=dataset_id, 
    #         dst=src
    #     )

    # copy_dataset_jsons(
    #     src=Path(os.environ['nnUNet_preprocessed']), 
    #     dst=src
    # )

    ##########################################

    from train import LABEL_SUBSETS

    all_labels = [LABEL_SUBSETS[ds_name] for ds_name in LABEL_SUBSETS.keys()]
    all_labels = np.unique([xx for x in all_labels for xx in x])
    # all_labels = {l: i+1 for i, l in enumerate(all_labels)}
    n_cls = len(all_labels)+1

    # ########################
    # # central
    # ########################

    train_dataset_ids = [2,4,5,6,7,9,14,37]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()
    for m in model.modules():
        if type(m) == nn.Dropout2d:
            m.p = 0.2

    ckpt = torch.load('checkpoints/central.pt')['model']
    model.load_state_dict(ckpt)

    exp_name = 'central_subset'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )
    
    # ########################
    # # Federated
    # ########################

    train_dataset_ids = [2,4,5,6,7,9,14,37]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    for m in model.modules():
        if type(m) == nn.Dropout2d:
            m.p = 0.2

    ckpt_root = Path('checkpoints/federated')
    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)
    
    exp_name = 'federated_subset'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    ########################
    # VSC
    ########################

    src = Path('/mnt/sds/sd20i001/malte/data/CT20K/nnUNet_preprocessed_batches')

    train_dataset_ids = [2]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/visceral_sc')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset002_visceral_sc'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    ########################
    # LiTS
    ########################

    train_dataset_ids = [4]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/lits')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset004_lits'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    # ########################
    # # BCV Abdomen
    # ########################

    train_dataset_ids = [5]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/bcv_abdomen')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset005_bcv_abdomen'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    ########################
    # BCV Cervix
    ########################

    train_dataset_ids = [6]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/bcv_cervix')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset006_bcv_cervix'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    ########################
    # CHAOS
    ########################

    train_dataset_ids = [7]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/chaos')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset007_chaos'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    ########################
    # AbdomenCT-1k
    ########################

    train_dataset_ids = [9]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/abdomenct1k')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset009_abdomenct1k'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    ########################
    # Learn2Reg
    ########################

    train_dataset_ids = [14]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('/mnt/sds/sd20i001/malte/code/fed-bayesian-avg/experiments/nnUNet/federated/learn2reg/1704556012.758104')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset014_learn2reg'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )

    # ########################
    # # TotalSegmentator
    # ########################

    train_dataset_ids = [37]
    test_dataset_ids = [2,4,5,6,7,9,14,37,1,38]
    train_dataset_names = [x.name for x in src.iterdir() if int(x.name[7:10]) in train_dataset_ids]
    model = Model(batch_size=1, dataset_names=train_dataset_names, LABEL_SUBSETS=LABEL_SUBSETS, trainer_name='FednnUNetTrainer').cuda()

    ckpt_root = Path('checkpoints/totalsegmentator')

    backbone_ckpt = torch.load(ckpt_root / 'ckpt.pt')
    # backbone_ckpt = {k.replace('all_modules.2', 'all_modules.1'): v for k, v in backbone_ckpt.items()}
    model.backbone.load_state_dict(backbone_ckpt)

    for train_ds_name in train_dataset_names:
        cls_head_ckpt = torch.load(ckpt_root / f'ckpt_{train_ds_name}.pt')
        model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

    exp_name = 'Dataset037_totalsegmentator'
    test(
        exp_name=exp_name,
        LABEL_SUBSETS=LABEL_SUBSETS,
        model=model,
        train_dataset_ids=train_dataset_ids,
        test_dataset_ids=test_dataset_ids,
        src=src
    )