# os.environ['nnUNet_results'] = '/mnt/hdd/data/CT20K/ct_ready_nnunet/results'
from typing import Union, Optional
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import SimpleITK as sitk
import torch
# from torch import autocast
import torch.nn as nn
from monai.metrics import HausdorffDistanceMetric
from aggregate import FedAVG
from utils import (
    uncertainty, 
    merged_logits_from_sep_cls, 
    merged_logits_from_sep_cls_with_uncert,
    onehot,
    hard_dice_score
)


LABEL_SUBSETS = {
    "Dataset001_visceral_gc": ["spleen", "right kidney", "left kidney", "liver", "aorta", "pancreas", "trachea", "urinary bladder"],
    "Dataset002_visceral_sc": ["spleen", "right kidney", "left kidney", "liver", "aorta", "pancreas", "trachea", "urinary bladder"],
    "Dataset004_lits": ["liver"],
    "Dataset005_bcv_abdomen": ["spleen", "right kidney", "left kidney", "liver", "stomach", "aorta", "pancreas"], # "inferior vena cava", 
    "Dataset006_bcv_cervix": ["urinary bladder"],
    "Dataset007_chaos": ["liver"],
    # "Dataset008_ctorg": ["liver", "urinary bladder"],
    "Dataset009_abdomenct1k": ["spleen", "liver", "pancreas"],
    "Dataset014_learn2reg": ["liver", "spleen"], # "urinary bladder"],
    # "Dataset018_sliver07": ["liver"],
    # "Dataset034_empire": ["liver"],
    "Dataset037_totalsegmentator": ['aorta', 'duodenum', 'esophagous', 'gall bladder', 'left kidney', 'pancreas', 'right kidney', 'stomach', 'trachea', 'urinary bladder'], # 'inferior vena cava', 'liver', 'spleen', 
    "Dataset038_amos": ["spleen", "right kidney", "left kidney", "gall bladder", "liver", "stomach", "aorta", "pancreas", "esophagous", "duodenum"],
}

N_ITERATIONS_PER_FED_EPOCH = {
    "Dataset001_visceral_gc": 150,
    "Dataset002_visceral_sc": 150,
    "Dataset004_lits": 50,
    "Dataset005_bcv_abdomen": 150, # "inferior vena cava", 
    "Dataset006_bcv_cervix": 150,
    "Dataset007_chaos": 50,
    # "Dataset008_ctorg": ["liver", "urinary bladder"],
    "Dataset009_abdomenct1k":150,
    "Dataset014_learn2reg": 50,
    # "Dataset018_sliver07": ["liver"],
    # "Dataset034_empire": ["liver"],
    "Dataset037_totalsegmentator": 300, # 'inferior vena cava', 'liver', 'spleen', 
    "Dataset038_amos": 300,
}

samples_per_dataset = {
    "Dataset001_visceral_gc": 40,
    "Dataset002_visceral_sc": 127, # actually less but we increase weight because of number of labels
    "Dataset004_lits": 126,
    "Dataset005_bcv_abdomen": 30, # actually less but we increase weight because of number of labels # "inferior vena cava", 
    "Dataset006_bcv_cervix": 30,
    "Dataset007_chaos": 20,
    # "Dataset008_ctorg": ["liver", "urinary bladder"],
    "Dataset009_abdomenct1k": 1000,
    "Dataset014_learn2reg": 8,
    # "Dataset018_sliver07": ["liver"],
    # "Dataset034_empire": ["liver"],
    "Dataset037_totalsegmentator": 1072, # 'inferior vena cava', 'liver', 'spleen', 
    "Dataset038_amos": 200,
}
# samples_per_dataset = {ds: n * len(LABEL_SUBSETS[ds]) for ds, n in samples_per_dataset.items()}

# LABEL_SUBSETS = {
#     'Dataset002_visceral_sc': ['spleen','right kidney','left kidney','liver','aorta','pancreas','right adrenal gland','left adrenal gland','trachea','urinary bladder','gallbladder','right lung','left lung','sternum','thyroid gland','first lumbar vertebrae','right psoas major','left psoas major','right rectus abdominis','left rectus abdominis'],
#     'Dataset004_lits': ['liver', 'lesion'],
#     'Dataset005_bcv_abdomen': ['spleen','right kidney','left kidney','liver','stomach','aorta','inferior vena cava','pancreas','right adrenal gland','left adrenal gland','gallbladder','esophagus','portal and splenic_vein'],
#     'Dataset006_bcv_cervix': ['urinary bladder','uterus','rectum','small bowel'],
#     'Dataset007_chaos': ['liver'],
#     'Dataset009_abdomenct1k': ['spleen','liver','pancreas','kidney'],
#     'Dataset014_learn2reg': ['liver','urinary bladder','lungs','kidneys'],
#     'Dataset037_totalsegmentator': ['spleen','right kidney','left kidney','gall bladder','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland','lung upper lobe_left','lung lower lobe left','lung upper lobe right','lung middle lobe right','lung lower lobe right','vertebrae_L2','vertebrae_L1','vertebrae_T12','vertebrae_T11','vertebrae_T10','vertebrae_T9','vertebrae_T8','vertebrae_T7','vertebrae_T6','vertebrae_T5','vertebrae_T4','vertebrae_T3','vertebrae_T2','vertebrae_T1','vertebrae_C7','vertebrae_C6','esophagous','trachea','heart_myocardium','heart_atrium_left','heart_ventricle_left','heart_atrium_right','heart_ventricle_right','pulmonary_artery','small_bowel','duodenum','colon','rib_left_1','rib_left_2','rib_left_3','rib_left_4','rib_left_5','rib_left_6','rib_left_7','rib_left_8','rib_left_9','rib_left_10','rib_left_11','rib_left_12','rib_right_1','rib_right_2','rib_right_3','rib_right_4','rib_right_5','rib_right_6','rib_right_7','rib_right_8','rib_right_9','rib_right_10','rib_right_11','rib_right_12','humerus_left','humerus_right','scapula_left','scapula_right','clavicula_left','clavicula_right','autochthon_left','autochthon_right','iliopsoas_left','iliopsoas_right','femur_left','vertebrae_L5','vertebrae_L4','vertebrae_L3','iliac_artery_left','iliac_artery_right','iliac_vena_left','iliac_vena_right','femur_right','hip_left','hip_right','sacrum','gluteus_maximus_left','gluteus_maximus_right','gluteus_medius_left','gluteus_medius_right','gluteus_minimus_left','gluteus_minimus_right','urinary bladder','brain','vertebrae_C5','vertebrae_C4','vertebrae_C3','vertebrae_C2','vertebrae_C1','face']

    # 'Dataset001_visceral_gc': ['spleen','right kidney','left kidney','liver','aorta','pancreas','right adrenal gland','left adrenal gland','trachea','urinary bladder','gallbladder','right lung','left lung','sternum','thyroid gland','first lumbar vertebrae','right psoas major','left psoas major','right rectus abdominis','left rectus abdominis'],
    # 'Dataset038_amos': ['spleen','right kidney','left kidney','gall bladder','liver','stomach','aorta','pancreas','right adrenal gland','left adrenal gland','esophagous','duodenum','postcava','bladder','prostate/uterus']
# }


def get_trainer(
        dataset_name_or_id: Union[int, str],
        configuration: str,
        fold: int,
        trainer_name: str = 'nnUNetTrainer',
        # nnunet_trainer_class,
        fedprox_mu: Optional[float] = None,
        plans_identifier: str = 'nnUNetPlans',
        use_compressed: bool = False,
        device: torch.device = torch.device('cuda'),
        *args,
        **kwargs
):
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from condistfl import ConDistFLTrainer
    from fed_trainer import FednnUNetTrainer, FednnUNetTrainerMarginalLoss
    # import src.trainers as trainers
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    trainers = {
        'FednnUNetTrainer': FednnUNetTrainer, # nnUNetTrainer
        'ConDistFLTrainer': ConDistFLTrainer,
        'FednnUNetTrainerMarginalLoss': FednnUNetTrainerMarginalLoss
    }
    nnunet_trainer_class = trainers[trainer_name]
    # nnunet_trainer_class = nnUNetTrainer # getattr(trainers, trainer_name)
    nnunet_trainer = nnunet_trainer_class(
        plans=plans, configuration=configuration, fold=fold,
        dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device,
        fedprox_mu=fedprox_mu,
        *args, **kwargs
    )
    return nnunet_trainer


def configure_trainers(
        # config,
        configuration,
        trainer_name,
        fold,
        LABELS,
        dropout_p=None,
        fedprox_mu=None,
        ckpt=None,
        batch_size=16,
        # *args, **kwargs
):
    # configuration = config['configuration']
    device = torch.device('cuda')
    get_trainer_fed = lambda dataset_name_or_id, *args, **kwargs: get_trainer(
        trainer_name=trainer_name, # config['trainer_name'],
        dataset_name_or_id=dataset_name_or_id, 
        configuration=configuration, 
        fold=fold, # config['fold'],
        fedprox_mu=fedprox_mu,
        plans_identifier='nnUNetPlans', 
        use_compressed=False, 
        device=device,
        *args, **kwargs
    )
    if trainer_name == 'FednnUNetTrainer':
        nnunet_trainers = [get_trainer_fed(d) for d in LABELS.keys()]
        master_trainer = get_trainer_fed(dataset_name_or_id='Dataset100_CT20K')
    elif trainer_name == 'ConDistFLTrainer':
        all_labels = [LABEL_SUBSETS[ds_name] for ds_name in LABELS.keys()]
        all_labels = np.unique([xx for x in all_labels for xx in x])
        all_labels = {l: i+1 for i, l in enumerate(all_labels)}
        n_cls = len(all_labels)+1

        nnunet_trainers = []
        for ds_name, ds_labels in LABELS.items():
            ds_labels = ds_labels['wanted_labels_in_dataset']
            foreground = np.sort([all_labels[l] for l in ds_labels]).tolist()
            background = [i for i in range(n_cls) if i not in foreground]
            trainer = get_trainer_fed(ds_name, foreground=foreground, background=background, num_classes=n_cls)
            nnunet_trainers.append(trainer)

        master_trainer = get_trainer_fed(dataset_name_or_id='Dataset100_CT20K', foreground=[i for i in range(1, n_cls)], background=[0])
    elif trainer_name == 'FednnUNetTrainerMarginalLoss':
        all_labels = [LABEL_SUBSETS[ds_name] for ds_name in LABELS.keys()]
        all_labels = np.unique([xx for x in all_labels for xx in x])
        all_labels = {l: i+1 for i, l in enumerate(all_labels)}
        n_cls = len(all_labels)+1

        nnunet_trainers = []
        for ds_name, ds_labels in LABELS.items():
            ds_labels = ds_labels['wanted_labels_in_dataset']
            foreground = np.sort([all_labels[l] for l in ds_labels]).tolist()
            trainer = get_trainer_fed(ds_name, foreground=foreground)
            nnunet_trainers.append(trainer)

        master_trainer = get_trainer_fed(dataset_name_or_id='Dataset100_CT20K', foreground=[i for i in range(1, n_cls)])
    
    # vi_kwargs = config.get('vi_kwargs', None)
    if dropout_p is not None:
        master_trainer.configuration_manager.network_arch_init_kwargs['dropout_op'] = f'torch.nn.Dropout{configuration[:2]}'
        master_trainer.configuration_manager.network_arch_init_kwargs['dropout_op_kwargs'] = {'p': dropout_p}
        # master_trainer.plans_manager.dropout_op = getattr(nn, f'Dropout{configuration[:2]}')
        # master_trainer.plans_manager.dropout_op_kwargs = {'p': dropout_p}
    master_trainer.initialize()

    conv_op = getattr(nn, f'Conv{configuration[:2]}')
    if trainer_name == 'ConDistFLTrainer' or trainer_name == 'FednnUNetTrainerMarginalLoss':
        master_trainer.network.decoder.seg_layers = nn.ModuleList([
            conv_op(m.in_channels, n_cls, m.kernel_size, stride=m.stride)
            for m in master_trainer.network.decoder.seg_layers
        ]).to(device)

    if ckpt is not None:
        n_epochs_trained = np.max([int(x.name.replace('.pt', '').split('_')[-1]) for x in exp_path.glob('ckpt_*.pt')])
        master_ckpt = torch.load(exp_path / f'ckpt_{n_epochs_trained}.pt', map_location='cpu')
        sd = master_trainer.network.state_dict()
        # master_ckpt = {k.replace('all_modules.2.weight', 'all_modules.1.weight').replace('all_modules.2.bias', 'all_modules.1.bias'): v for k, v in master_ckpt.items()} # if k.startswith('encoder') or k.startswith('decoder')}
        for k, v in sd.items():
            if k not in master_ckpt:
                # print(k)
                master_ckpt[k] = v
        master_trainer.network.load_state_dict(master_ckpt)
        del master_ckpt
    for t in nnunet_trainers:
        t.batch_size = batch_size # master_trainer.batch_size
        if dropout_p is not None:
        # if vi_kwargs is not None:
            t.configuration_manager.network_arch_init_kwargs['dropout_op'] = f'torch.nn.Dropout{configuration[:2]}'
            t.configuration_manager.network_arch_init_kwargs['dropout_op_kwargs'] = {'p': dropout_p}
        t.configuration_manager.configuration['patch_size'] = master_trainer.configuration_manager.patch_size
        t.configuration_manager.configuration['batch_dice'] = master_trainer.configuration_manager.batch_dice
        t.on_train_start()
        t.network = deepcopy(master_trainer.network)
        # t.network.encoder = deepcopy(master_trainer.network.encoder)
        # t.network.decoder.transpconvs = deepcopy(master_trainer.network.decoder.transpconvs)
        # t.network.decoder.stages = deepcopy(master_trainer.network.decoder.stages)
        
        dataset_name = t.plans_manager.dataset_name

        from transforms import LabelTransformNNUnet
        ds_labels = LABELS[dataset_name]
        lt = LabelTransformNNUnet(**ds_labels)
        t.dataloader_train.transform.transforms.append(lt)
        t.dataloader_val.transform.transforms.append(lt)

        if trainer_name == 'FednnUNetTrainer':
            n_seg_heads = len(ds_labels['wanted_labels_in_dataset']) + 1            
            t.network.decoder.seg_layers = nn.ModuleList([
                conv_op(m.in_channels, n_seg_heads, m.kernel_size, stride=m.stride)
                for m in t.network.decoder.seg_layers
            ]).to(device)
        elif trainer_name == 'ConDistFLTrainer' or trainer_name == 'FednnUNetTrainerMarginalLoss':
            t.network.decoder.seg_layers = nn.ModuleList([
                conv_op(m.in_channels, n_cls, m.kernel_size, stride=m.stride)
                for m in t.network.decoder.seg_layers
            ]).to(device)

            from condistfl import ConDistFLLabelTransform
            ds_labels = LABELS[t.plans_manager.dataset_name]['wanted_labels_in_dataset']
            cdt = ConDistFLLabelTransform(ds_labels=ds_labels, all_labels=all_labels)
            t.dataloader_train.transform.transforms.append(cdt)
            t.dataloader_val.transform.transforms.append(cdt)

        t.optimizer, t.lr_scheduler = t.configure_optimizers()

        if ckpt is not None:
            try:
                ckpt_path = ckpt / dataset_name / f'{trainer_name}__nnUNetPlans__{configuration}' / f'fold_{fold}' / 'checkpoint_latest.pth'
                t.load_checkpoint(str(ckpt_path))
            except:
                ckpt_path = ckpt / dataset_name / f'{trainer_name}__nnUNetPlans__{configuration}' / f'fold_{fold}' / 'checkpoint_final.pth'
                t.load_checkpoint(str(ckpt_path))

            t_ckpt = torch.load(exp_path / f'ckpt_{t.plans_manager.dataset_name}_{n_epochs_trained}.pt')
            # t_ckpt = {f'decoder.seg_layers.{n}': w for n, w in t_ckpt.items()}
            t.network.decoder.seg_layers.load_state_dict(t_ckpt)
            del t_ckpt
    
    return master_trainer, nnunet_trainers


def train_all(
        dataset_ids,
        configuration, 
        trainer_name, 
        fold,
        dropout_p,
        # config,
        exp_path,
        from_ckpt=False,
        subset=True
):
    
    from nnunetv2.paths import nnUNet_raw

    LABELS = {d: l for d, l in LABEL_SUBSETS.items() if int(d.split('_')[0].replace('Dataset','')) in dataset_ids}

    LABELS = {
        dataset_name: {
            'wanted_labels_in_dataset': {l: i+1 for i, l in enumerate(labels)}
        } for dataset_name, labels in LABELS.items()
    }

    for dataset_name in LABELS.keys():
        with open(Path(nnUNet_raw) / dataset_name / 'dataset.json', 'r') as f:
            labels = json.load(f)['labels']
        LABELS[dataset_name]['all_labels_in_dataset'] = {int(v): k for k, v in labels.items()}
        if not subset:
            LABELS[dataset_name]['wanted_labels_in_dataset'] = {v: k for k, v in LABELS[dataset_name]['all_labels_in_dataset'].items()}

    config = {
        'exp_name': exp_path.parents[1].name,
        'trainer_name': trainer_name,
        'configuration': configuration,
        'fold': fold,
        'dropout_p': dropout_p,
        'dataset_ids': dataset_ids
    }
    with open(exp_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(exp_path / 'LABELS.json', 'w') as f:
        json.dump(LABELS, f, indent=4)

    batch_size = 16
    # ckpt = exp_path if from_ckpt else None
    master_trainer, nnunet_trainers = configure_trainers(
        configuration=configuration,
        trainer_name=trainer_name,
        fold=fold,
        dropout_p=dropout_p, 
        LABELS=LABELS,
        ckpt=None, # ckpt,
        batch_size=batch_size
    )

    # from nnunetv2.paths import nnUNet_raw
    # from nnunetv2.run.run_training import get_trainer_from_args
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
    from nnunetv2.utilities.helpers import dummy_context

    master_trainer.network.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in range(len(master_trainer.network.decoder.seg_layers))])
    master_trainer.network.decoder.deep_supervision = True
    # backbone = deepcopy(master_trainer.network)
    # cls_heads = []

    loss = master_trainer._build_loss()
    # master_optim, lr_scheduler = master_trainer.configure_optimizers()
    device = master_trainer.device
    for t in nnunet_trainers:
        t.configuration_manager.configuration['patch_size'] = master_trainer.configuration_manager.patch_size
        t.configuration_manager.configuration['batch_dice'] = master_trainer.configuration_manager.batch_dice
        # cls_heads.append(deepcopy(t.network.decoder.seg_layers))
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = deepcopy(master_trainer.network)
            self.cls_heads = nn.ModuleDict({
                t.plans_manager.dataset_name: deepcopy(t.network.decoder.seg_layers) for t in nnunet_trainers
            })
        
        def forward(self, x):
            out = self.backbone(x)
            outs = {n: [m(o) for m, o in zip(cls_head[::-1], out)] for i, (n, cls_head) in enumerate(self.cls_heads.items())}
            return outs
    
    model = Model()
    # ckpt = torch.load('experiments/nnUNet/all/subset_restart/1715081545.010999/ckpt_65.pt')
    # ckpt = torch.load('experiments/nnUNet/all/subset_restart/1715149516.556817/ckpt_50.pt')
    # model.load_state_dict(ckpt['model'])
    optimizer = torch.optim.SGD(model.parameters(), master_trainer.initial_lr, weight_decay=master_trainer.weight_decay,
                                momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, master_trainer.initial_lr, master_trainer.num_epochs)
    # grad_scaler = deepcopy(master_trainer.grad_scaler)
    if from_ckpt:
        n_epochs_trained = np.max([int(x.name.replace('.pt', '').split('_')[-1]) for x in exp_path.glob('ckpt_*.pt')])
        master_ckpt = torch.load(exp_path / f'ckpt_{n_epochs_trained}.pt', map_location='cpu')
        model.load_state_dict(master_ckpt['model'])
        optimizer.load_state_dict(master_ckpt['optimizer'])
        lr_scheduler.load_state_dict(master_ckpt['lr_scheduler'])
        master_trainer.current_epoch = n_epochs_trained

        for t in nnunet_trainers:
            t.load_checkpoint(str(exp_path / t.plans_manager.dataset_name / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_1' / 'checkpoint_latest.pth'))

    for epoch in range(master_trainer.current_epoch, master_trainer.num_epochs):
        model.train()
        for t in nnunet_trainers:
            t.on_epoch_start()
            t.on_train_epoch_start()
        train_outputs = {t.plans_manager.dataset_name: [] for t in nnunet_trainers}
        for batch_id in range(250): # master_trainer.num_iterations_per_epoch):
            for t in nnunet_trainers:
                batch = next(t.dataloader_train)
                t_data = batch['data'].to(device)
                t_target = batch['target']
                t_target = [i.to(device, non_blocking=True) for i in t_target]
                optimizer.zero_grad()
                output = model(t_data)
                output = output[t.plans_manager.dataset_name]
                l = loss(output, t_target)
                l.backward()
                optimizer.step()
                train_outputs[t.plans_manager.dataset_name].append({'loss': l.detach().cpu().numpy()})
        
        # lr_scheduler.step(epoch)
        for t in nnunet_trainers:
            t.on_train_epoch_end(train_outputs[t.plans_manager.dataset_name])
        
        val_outputs = {t.plans_manager.dataset_name: [] for t in nnunet_trainers}
        model.eval()
        for batch_id in range(master_trainer.num_val_iterations_per_epoch):
            for t in nnunet_trainers:
                batch = next(t.dataloader_val)
                t_data = batch['data'].to(device)
                t_target = batch['target']
                t_target = [i.to(device, non_blocking=True) for i in t_target]

                with torch.no_grad():
                    output = model(t_data)
                output = output[t.plans_manager.dataset_name]
                l = loss(output, t_target)

                output = output[0].detach()
                target = t_target[0].detach()

                # the following is needed for online evaluation. Fake dice (green line)
                axes = [0] + list(range(2, len(output.shape)))

                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg
                # predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
                mask = None

                tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
                
                tp_hard = tp.detach().cpu().numpy()
                fp_hard = fp.detach().cpu().numpy()
                fn_hard = fn.detach().cpu().numpy()

                # [1:] in order to remove background
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
                # print(l.item(), tp_hard, t.plans_manager.dataset_name, 'test')
                val_outputs[t.plans_manager.dataset_name].append({'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard})

        for t in nnunet_trainers:
        #     print(t.plans_manager.dataset_name)
            t.on_validation_epoch_end(val_outputs[t.plans_manager.dataset_name])
            t.on_epoch_end()
        
        # _val_outputs = [xx for x in val_outputs.values() for xx in x]
        # master_trainer.on_validation_epoch_end([{'loss': np.array([0.]), 'tp_hard': np.array([0.]), 'fp_hard': np.array([1.]), 'fn_hard': np.array([0.])}])
        # master_trainer.on_epoch_end()
        train_metrics = {t.plans_manager.dataset_name: t.logger.my_fantastic_logging for t in nnunet_trainers}
        torch.save(train_metrics, exp_path / 'train_metrics.pt')
        if epoch % 5 == 0:
            ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), lr_scheduler=lr_scheduler.state_dict())
            torch.save(ckpt, exp_path / f'ckpt_{epoch}.pt')
            # torch.save(master_trainer.network.state_dict(), exp_path / f'ckpt_e{epoch}.pt')
            # for t in nnunet_trainers:
            #     torch.save(t.network.state_dict(), exp_path / f'ckpt_e{epoch}_{t.plans_manager.dataset_name}.pt')
    for t in nnunet_trainers:
        t.on_train_end()
    # master_trainer.on_train_end()


def train_all_marginal(
        dataset_ids,
        configuration, 
        trainer_name, 
        fold,
        dropout_p,
        # config,
        exp_path,
        from_ckpt=False,
        subset=True
):
    
    from nnunetv2.paths import nnUNet_raw

    LABELS = {d: l for d, l in LABEL_SUBSETS.items() if int(d.split('_')[0].replace('Dataset','')) in dataset_ids}

    LABELS = {
        dataset_name: {
            'wanted_labels_in_dataset': {l: i+1 for i, l in enumerate(labels)}
        } for dataset_name, labels in LABELS.items()
    }

    for dataset_name in LABELS.keys():
        with open(Path(nnUNet_raw) / dataset_name / 'dataset.json', 'r') as f:
            labels = json.load(f)['labels']
        LABELS[dataset_name]['all_labels_in_dataset'] = {int(v): k for k, v in labels.items()}
        if not subset:
            LABELS[dataset_name]['wanted_labels_in_dataset'] = {v: k for k, v in LABELS[dataset_name]['all_labels_in_dataset'].items()}

    config = {
        'exp_name': exp_path.parents[1].name,
        'trainer_name': trainer_name,
        'configuration': configuration,
        'fold': fold,
        'dropout_p': dropout_p,
        'dataset_ids': dataset_ids
    }
    with open(exp_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(exp_path / 'LABELS.json', 'w') as f:
        json.dump(LABELS, f, indent=4)

    batch_size = 4
    # ckpt = exp_path if from_ckpt else None
    master_trainer, nnunet_trainers = configure_trainers(
        configuration=configuration,
        trainer_name=trainer_name,
        fold=fold,
        dropout_p=dropout_p, 
        LABELS=LABELS,
        ckpt=None, # ckpt,
        batch_size=batch_size
    )

    # from nnunetv2.paths import nnUNet_raw
    # from nnunetv2.run.run_training import get_trainer_from_args
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
    from nnunetv2.utilities.helpers import dummy_context

    # master_trainer.network.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in range(len(master_trainer.network.decoder.seg_layers))])
    master_trainer.network.decoder.deep_supervision = True
    # backbone = deepcopy(master_trainer.network)
    # cls_heads = []

    loss = master_trainer._build_loss()
    # master_optim, lr_scheduler = master_trainer.configure_optimizers()
    device = master_trainer.device
    losses = {}
    for t in nnunet_trainers:
        t.configuration_manager.configuration['patch_size'] = master_trainer.configuration_manager.patch_size
        t.configuration_manager.configuration['batch_dice'] = master_trainer.configuration_manager.batch_dice
        losses[t.plans_manager.dataset_name] = t._build_loss()
        # cls_heads.append(deepcopy(t.network.decoder.seg_layers))
    
    model = master_trainer.network

    optimizer = torch.optim.SGD(model.parameters(), master_trainer.initial_lr, weight_decay=master_trainer.weight_decay,
                                momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, master_trainer.initial_lr, master_trainer.num_epochs)
    # grad_scaler = deepcopy(master_trainer.grad_scaler)
    
    if from_ckpt:
        n_epochs_trained = np.max([int(x.name.replace('.pt', '').split('_')[-1]) for x in exp_path.glob('ckpt_*.pt')])
        master_ckpt = torch.load(exp_path / f'ckpt_{n_epochs_trained}.pt', map_location='cpu')
        model.load_state_dict(master_ckpt['model'])
        optimizer.load_state_dict(master_ckpt['optimizer'])
        lr_scheduler.load_state_dict(master_ckpt['lr_scheduler'])
        master_trainer.current_epoch = n_epochs_trained

        for t in nnunet_trainers:
            t.load_checkpoint(str(exp_path / t.plans_manager.dataset_name / f'{trainer_name}__nnUNetPlans__2d' / 'fold_1' / 'checkpoint_best.pth'))
            # needed with chkpt best!?
            # t.current_epoch = n_epochs_trained
    
    for epoch in range(master_trainer.current_epoch, master_trainer.num_epochs):
        model.train()
        for t in nnunet_trainers:
            trt = deepcopy(t.dataloader_train.transform.transforms)
            valt = deepcopy(t.dataloader_val.transform.transforms)
            # otherwise processes crash da fu**
            t.dataloader_train, t.dataloader_val = t.get_dataloaders()
            t.dataloader_train.transform.transforms = trt
            t.dataloader_val.transform.transforms = valt
            t.on_epoch_start()
            t.on_train_epoch_start()
        train_outputs = {t.plans_manager.dataset_name: [] for t in nnunet_trainers}
        for batch_id in range(1000): # master_trainer.num_iterations_per_epoch):
            for t in nnunet_trainers:
                batch = next(t.dataloader_train)
                t_data = batch['data'].to(device)
                t_target = batch['target']
                t_target = [i.to(device, non_blocking=True) for i in t_target]
                optimizer.zero_grad()
                output = model(t_data)
                # output = output[t.plans_manager.dataset_name]
                # l = loss(output, t_target)
                l = losses[t.plans_manager.dataset_name](output, t_target)
                l.backward()
                optimizer.step()
                train_outputs[t.plans_manager.dataset_name].append({'loss': l.detach().cpu().numpy()})
        
        # lr_scheduler.step(epoch)
        for t in nnunet_trainers:
            t.on_train_epoch_end(train_outputs[t.plans_manager.dataset_name])
        
        val_outputs = {t.plans_manager.dataset_name: [] for t in nnunet_trainers}
        model.eval()
        for batch_id in range(master_trainer.num_val_iterations_per_epoch):
            for t in nnunet_trainers:
                batch = next(t.dataloader_val)
                t_data = batch['data'].to(device)
                t_target = batch['target']
                t_target = [i.to(device, non_blocking=True) for i in t_target]

                with torch.no_grad():
                    output = model(t_data)
                # output = output[t.plans_manager.dataset_name]
                # l = loss(output, t_target)
                l = losses[t.plans_manager.dataset_name](output, t_target)

                output = output[0].detach()
                target = t_target[0].detach()

                # the following is needed for online evaluation. Fake dice (green line)
                axes = [0] + list(range(2, len(output.shape)))

                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg
                # predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
                mask = None

                tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
                
                tp_hard = tp.detach().cpu().numpy()
                fp_hard = fp.detach().cpu().numpy()
                fn_hard = fn.detach().cpu().numpy()

                # [1:] in order to remove background
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
                # print(l.item(), tp_hard, t.plans_manager.dataset_name, 'test')
                val_outputs[t.plans_manager.dataset_name].append({'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard})

        for t in nnunet_trainers:
        #     print(t.plans_manager.dataset_name)
            t.on_validation_epoch_end(val_outputs[t.plans_manager.dataset_name])
            t.on_epoch_end()
        
        # _val_outputs = [xx for x in val_outputs.values() for xx in x]
        # master_trainer.on_validation_epoch_end([{'loss': np.array([0.]), 'tp_hard': np.array([0.]), 'fp_hard': np.array([1.]), 'fn_hard': np.array([0.])}])
        # master_trainer.on_epoch_end()
        train_metrics = {t.plans_manager.dataset_name: t.logger.my_fantastic_logging for t in nnunet_trainers}
        torch.save(train_metrics, exp_path / 'train_metrics.pt')
        if epoch % 5 == 0:
            ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), lr_scheduler=lr_scheduler.state_dict())
            torch.save(ckpt, exp_path / f'ckpt_{epoch}.pt')
            # torch.save(master_trainer.network.state_dict(), exp_path / f'ckpt_e{epoch}.pt')
            # for t in nnunet_trainers:
            #     torch.save(t.network.state_dict(), exp_path / f'ckpt_e{epoch}_{t.plans_manager.dataset_name}.pt')
    for t in nnunet_trainers:
        t.on_train_end()
    # master_trainer.on_train_end()


def train_single(
        dataset_name_or_id, 
        configuration='2d', 
        fold='all', 
        trainer_name='nnUNetTrainer',
        num_epochs=100,
        # num_burn_in_steps=5000,
        # norm_sigma=200.,
        # save_interval=100,
        # save_epoch_start=40
        mcmc_kwargs=None,
        vi_kwargs=None
):
    from torch.backends import cudnn

    # from nnunetv2.run.run_training import run_training
    # from nnunetv2.run.run_training import get_trainer_from_args
    # run_training(dataset_name_or_id, configuration=configuration, fold=fold, trainer_class_name=trainer_name)
    plans_identifier = 'nnUNetPlans'
    pretrained_weights = None
    num_gpus = 1
    use_compressed_data = False
    export_validation_probabilities = False
    continue_training = False
    only_run_validation = False
    disable_checkpointing = False
    device = torch.device('cuda')
    # nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_name,
    #                                         plans_identifier, use_compressed_data, device=device)
    # nnunet_trainer_class = getattr(trainers, trainer_name)
    nnunet_trainer = get_trainer(
        trainer_name=trainer_name,
        dataset_name_or_id=dataset_name_or_id, 
        configuration=configuration, 
        fold=fold,
        plans_identifier=plans_identifier, 
        use_compressed=use_compressed_data, 
        device=device,
    )
   
    # nnunet_trainer.num_epochs = num_epochs
    if mcmc_kwargs is not None:
        nnunet_trainer.optimizer_kwargs = mcmc_kwargs['optimizer_kwargs']
        nnunet_trainer.save_interval = mcmc_kwargs['save_interval']
        nnunet_trainer.save_epoch_start = mcmc_kwargs['save_epoch_start']
        nnunet_trainer.initial_lr = mcmc_kwargs.get('initial_lr', 1e-2)

    if vi_kwargs is not None:
        nnunet_trainer.plans_manager.dropout_op = getattr(nn, f'Dropout{configuration}')
        nnunet_trainer.plans_manager.dropout_op_kwargs = {'p': vi_kwargs['dropout_p']}

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True
    if not only_run_validation:
        nnunet_trainer.run_training()
    nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def train_federated_separate_heads(
        dataset_ids,
        configuration, 
        trainer_name, 
        fold,
        dropout_p,
        fedprox_mu,
        # config,
        exp_path,
        from_ckpt,
        subset=True
):
    from nnunetv2.paths import nnUNet_raw

    LABELS = {d: l for d, l in LABEL_SUBSETS.items() if int(d.split('_')[0].replace('Dataset','')) in dataset_ids}

    LABELS = {
        dataset_name: {
            'wanted_labels_in_dataset': {l: i+1 for i, l in enumerate(labels)}
        } for dataset_name, labels in LABELS.items()
    }

    for dataset_name in LABELS.keys():
        with open(Path(nnUNet_raw) / dataset_name / 'dataset.json', 'r') as f:
            labels = json.load(f)['labels']
        LABELS[dataset_name]['all_labels_in_dataset'] = {int(v): k for k, v in labels.items()}
        if not subset:
            LABELS[dataset_name]['wanted_labels_in_dataset'] = {v: k for k, v in LABELS[dataset_name]['all_labels_in_dataset'].items()}

    config = {
        'exp_name': exp_path.parents[1].name,
        'trainer_name': trainer_name,
        'configuration': configuration,
        'fold': fold,
        'dropout_p': dropout_p,
        'dataset_ids': dataset_ids
    }
    with open(exp_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(exp_path / 'LABELS.json', 'w') as f:
        json.dump(LABELS, f, indent=4)

    batch_size = 16
    ckpt = exp_path if from_ckpt else None
    master_trainer, nnunet_trainers = configure_trainers(
        configuration=configuration,
        trainer_name=trainer_name,
        fold=fold,
        dropout_p=dropout_p, 
        fedprox_mu=fedprox_mu,
        LABELS=LABELS,
        ckpt=ckpt,
        batch_size=batch_size
    )
    master_trainer.network.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in nnunet_trainers[0].network.decoder.seg_layers])
    
    ### on TS FINETUNED
    # master_ckpt = torch.load('experiments/nnUNet/federated/ts/1715065585.558396/ckpt_300.pt')
    # try:
    #     master_trainer.network.load_state_dict(master_ckpt)
    # except:
    #     import pdb;pdb.set_trace()
    # for t in nnunet_trainers:
    #     try:
    #         if t.plans_manager.dataset_name != 'Dataset037_totalsegmentator':
    #             cls_head_ckpt = torch.load(f'experiments/nnUNet/federated/subset/1711118671.498861/ckpt_{t.plans_manager.dataset_name}_900.pt')
    #         else:
    #             cls_head_ckpt = torch.load('experiments/nnUNet/federated/ts/1715065585.558396/ckpt_Dataset037_totalsegmentator_300.pt')
    #         t.network.decoder.seg_layers.load_state_dict(cls_head_ckpt)
    #     except:
    #         import pdb;pdb.set_trace()

    ### FINETUNE TS FED
    # master_ckpt = torch.load('experiments/nnUNet/federated/subset/1711118671.498861/ckpt_900.pt')
    # cls_head_ckpt = torch.load('experiments/nnUNet/federated/subset/1711118671.498861/ckpt_Dataset037_totalsegmentator_900.pt')
    # master_trainer.network.load_state_dict(master_ckpt)
    # nnunet_trainers[0].network.decoder.seg_layers.load_state_dict(cls_head_ckpt)

    ##### Continue FED
    # master_ckpt = torch.load('experiments/nnUNet/federated/subset/1711118671.498861/ckpt_900.pt')
    # master_ckpt = torch.load('experiments/nnUNet/federated/subset_continue/1715093009.562455/ckpt_280.pt')
    # master_trainer.network.load_state_dict(master_ckpt)
    # for t in nnunet_trainers:
    #     # cls_head_ckpt = torch.load(f'experiments/nnUNet/federated/subset/1711118671.498861/ckpt_{t.plans_manager.dataset_name}_900.pt')
    #     # cls_head_ckpt = torch.load(f'experiments/nnUNet/federated/subset_continue/1715093009.562455/ckpt_{t.plans_manager.dataset_name}_280.pt')
    #     t.network.decoder.seg_layers.load_state_dict(cls_head_ckpt)
    
    #     if t.plans_manager.dataset_name == 'Dataset037_totalsegmentator':
    #         t.loss.loss.ce.weight = torch.tensor([1.,1.,1.5,1.,4.,1.,1.,1.,1.,1.,1.]).cuda()

    # ### FINETUNE TS ALL
    # import pdb;pdb.set_trace()
    # ckpt = torch.load('experiments/nnUNet/all/subset/1714744995.155121/ckpt_235.pt')
    # # master_ckpt = torch.load('experiments/nnUNet/federated/subset/1711118671.498861/ckpt_900.pt')
    # # cls_head_ckpt = torch.load('experiments/nnUNet/federated/subset/1711118671.498861/ckpt_Dataset037_totalsegmentator_900.pt')
    # # master_trainer.network.load_state_dict(master_ckpt)
    # # nnunet_trainers[0].network.decoder.seg_layers.load_state_dict(cls_head_ckpt)

    # n_samples_per_client = np.array([len(list((Path(nnUNet_raw) / t.plans_manager.dataset_name / 'imagesTr').iterdir())) for t in nnunet_trainers])
    # import pdb;pdb.set_trace()
    # samples_per_dataset = {ds: n * len(LABEL_SUBSETS)[ds] for ds, n in samples_per_dataset.items()}
    n_samples_per_client = np.array([samples_per_dataset[t.plans_manager.dataset_name] for t in nnunet_trainers])
    # n_labels_per_client = np.array([len(t.dataset_json['labels']) for t in nnunet_trainers])
    client_weights_samples = n_samples_per_client / np.sum(n_samples_per_client)
    # client_weights_labels = n_labels_per_client / np.sum(n_labels_per_client)
    client_weights = client_weights_samples
    fed_avg = FedAVG(client_weights=client_weights.tolist())
    new_state_dict = master_trainer.network.state_dict()

    n_epochs = 2000
    if from_ckpt:
        m = torch.load(exp_path / 'train_metrics.pt')
        ds = list(LABELS.keys())[0]
        epochs_trained = len(m[ds]['train_losses'])
    else:
        epochs_trained = 0
    
    for epoch in range(epochs_trained, n_epochs):
        for t in nnunet_trainers:
            # print(t.plans_manager.dataset_name)
            cls_head_state_dict = {k: v for k, v in t.network.state_dict().items() if k.startswith('decoder.seg_layers')}
            for k, v in cls_head_state_dict.items():
                new_state_dict.update({k: v})
            t.network.load_state_dict(new_state_dict)
        state_dicts = []
        for t in nnunet_trainers:
            t.on_epoch_start()
            t.on_train_epoch_start()
            t.network.decoder.deep_supervision = True
            train_outputs = []
            for batch_id in range(t.num_iterations_per_epoch):
                batch = next(t.dataloader_train)
                loss = t.train_step(batch)
                train_outputs.append(loss)
                # train_outputs.append(t.train_step(next(t.dataloader_train)))
            t.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                t.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(t.num_val_iterations_per_epoch):
                    val_outputs.append(t.validation_step(next(t.dataloader_val)))
                t.on_validation_epoch_end(val_outputs)

            t.on_epoch_end()

            state_dict = {k: v for k, v in t.network.state_dict().items() if not k.startswith('decoder.seg_layers')}
            state_dicts.append(state_dict)

        new_state_dict = fed_avg(state_dicts)
        train_metrics = {t.plans_manager.dataset_name: t.logger.my_fantastic_logging for t in nnunet_trainers}
        torch.save(train_metrics, exp_path / 'train_metrics.pt')
        if epoch % 20 == 0:
            torch.save(new_state_dict, exp_path / f'ckpt_{epoch}.pt')
            for t in nnunet_trainers:
                torch.save(t.network.decoder.seg_layers.state_dict(), exp_path / f'ckpt_{t.plans_manager.dataset_name}_{epoch}.pt')
            
        t.on_train_end()


def train_federated_no_separate_heads(
        dataset_ids,
        configuration, 
        trainer_name, 
        fold,
        dropout_p,
        fedprox_mu,
        # config,
        exp_path,
        from_ckpt,
        subset=True
):
    from nnunetv2.paths import nnUNet_raw

    LABELS = {d: l for d, l in LABEL_SUBSETS.items() if int(d.split('_')[0].replace('Dataset','')) in dataset_ids}

    LABELS = {
        dataset_name: {
            'wanted_labels_in_dataset': {l: i+1 for i, l in enumerate(labels)}
        } for dataset_name, labels in LABELS.items()
    }

    for dataset_name in LABELS.keys():
        with open(Path(nnUNet_raw) / dataset_name / 'dataset.json', 'r') as f:
            labels = json.load(f)['labels']
        LABELS[dataset_name]['all_labels_in_dataset'] = {int(v): k for k, v in labels.items()}
        if not subset:
            LABELS[dataset_name]['wanted_labels_in_dataset'] = {v: k for k, v in LABELS[dataset_name]['all_labels_in_dataset'].items()}

    config = {
        'exp_name': exp_path.parents[1].name,
        'trainer_name': trainer_name,
        'configuration': configuration,
        'fold': fold,
        'dropout_p': dropout_p,
        'fedprox_mu': fedprox_mu,
        'dataset_ids': dataset_ids
    }
    with open(exp_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(exp_path / 'LABELS.json', 'w') as f:
        json.dump(LABELS, f, indent=4)

    batch_size = 4
    ckpt = exp_path if from_ckpt else None
    master_trainer, nnunet_trainers = configure_trainers(
        configuration=configuration,
        trainer_name=trainer_name,
        fold=fold,
        dropout_p=dropout_p, 
        fedprox_mu=fedprox_mu,
        LABELS=LABELS,
        ckpt=ckpt,
        batch_size=batch_size
    )
    
    for t in nnunet_trainers:
        t.num_iterations_per_epoch = 1000 # 250 * 16/4
        t.num_val_iterations_per_epoch = 200 # 50 * 16/4
    # master_trainer.network.decoder.seg_layers = nn.ModuleList([nn.Identity() for _ in nnunet_trainers[0].network.decoder.seg_layers])

    n_samples_per_client = np.array([samples_per_dataset[t.plans_manager.dataset_name] for t in nnunet_trainers])
    client_weights_samples = n_samples_per_client / np.sum(n_samples_per_client)
    client_weights = client_weights_samples

    fed_avg = FedAVG(client_weights=client_weights.tolist())
    new_state_dict = master_trainer.network.state_dict()

    n_epochs = 2000
    if from_ckpt:
        m = torch.load(exp_path / 'train_metrics.pt')
        ds = list(LABELS.keys())[0]
        epochs_trained = len(m[ds]['train_losses'])
    else:
        epochs_trained = 0

    for epoch in range(epochs_trained, n_epochs):
        # if new_state_dict is not None:

        for t in nnunet_trainers:
            t.network.load_state_dict(new_state_dict)
        state_dicts = []
        for t in nnunet_trainers:
            print(t.plans_manager.dataset_name)
            t.on_epoch_start()
            t.on_train_epoch_start()

            trt = deepcopy(t.dataloader_train.transform.transforms)
            valt = deepcopy(t.dataloader_val.transform.transforms)
            # otherwise processes crash da fu**
            t.dataloader_train, t.dataloader_val = t.get_dataloaders()
            t.dataloader_train.transform.transforms = trt
            t.dataloader_val.transform.transforms = valt

            t.network.decoder.deep_supervision = True
            train_outputs = []
            for batch_id in range(t.num_iterations_per_epoch):
                batch = next(t.dataloader_train)
                loss = t.train_step(batch)
                train_outputs.append(loss)
                # train_outputs.append(t.train_step(next(t.dataloader_train)))
            t.on_train_epoch_end(train_outputs)
            
            with torch.no_grad():
                t.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(t.num_val_iterations_per_epoch):
                    val_outputs.append(t.validation_step(next(t.dataloader_val)))
                t.on_validation_epoch_end(val_outputs)

            t.on_epoch_end()

            state_dict = t.network.state_dict() # {k: v for k, v in t.network.state_dict().items() if not k.startswith('decoder.seg_layers')}
            state_dicts.append(state_dict)

        new_state_dict = fed_avg(state_dicts)
        train_metrics = {t.plans_manager.dataset_name: t.logger.my_fantastic_logging for t in nnunet_trainers}
        torch.save(train_metrics, exp_path / 'train_metrics.pt')

        if epoch % 20 == 0:
            torch.save(new_state_dict, exp_path / f'ckpt_{epoch}.pt')
            # for t in nnunet_trainers:
            #     torch.save(t.network.decoder.seg_layers.state_dict(), exp_path / f'ckpt_{t.plans_manager.dataset_name}_{epoch}.pt')


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    import os

    from argparse import ArgumentParser
    import json
    import shutil
    parser = ArgumentParser()
    parser.add_argument('--mode', '-m', default='all')
    parser.add_argument('--exp_name', default='all_subset_marginal') # , default='visceral_gc')
    # parser.add_argument('--config', '-c', default='./configs/all_labels_cluster.json')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--nnUNet_raw', default='/gpfs/bwfor/work/ws/hd_cn265-fun/CT20K/ct_ready_nnunet/raw')# /gpfs/bwfor/work/ws/hd_cn265-fun
    parser.add_argument('--nnUNet_preprocessed', default='/gpfs/bwfor/work/ws/hd_cn265-fun/CT20K/ct_ready_nnunet/preprocessed_global2') # '../../data/CT20K/ct_ready_nnunet/preprocessed_global2')
    # parser.add_argument('--nnUNet_raw', default='../../data/CT20K/ct_ready_nnunet/raw')
    # parser.add_argument('--nnUNet_preprocessed', default='../../data/CT20K/ct_ready_nnunet/preprocessed_global2') # 
    parser.add_argument('--trainer_name', default='FednnUNetTrainerMarginalLoss')
    parser.add_argument('--configuration', default='2d')
    parser.add_argument('--fold', default=1)
    parser.add_argument('--dropout_p', default=0.3)
    parser.add_argument('--fedprox_mu', default=None, type=float)
    parser.add_argument('--dataset_ids', nargs='+', default=[2, 4, 5, 6, 7, 9, 14, 37], type=int) #        1,2,4,5,6,7,8,9,14,18,34,38
    parser.add_argument('--processes', default=0, type=int)
    parser.add_argument('--ts') # , default='1711118671.498861') # , default='1703153202.691494')
    parser.add_argument('--all', action='store_true', default=False)
    args = parser.parse_args()
    
    ts = str(datetime.now().timestamp()) if args.ts is None else args.ts
    
    print('*'*10)
    print(ts)
    print('*'*10)
    # with open(args.config, 'r') as f:
    #     config = json.load(f)
    
    os.environ['nnUNet_n_proc_DA'] = str(args.processes) # '0'

    os.environ['nnUNet_raw'] = args.nnUNet_raw
    os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
    exp_path = Path('./experiments/nnUNet') / args.mode / args.exp_name / ts
    if not exp_path.exists():
        exp_path.mkdir(parents=True)
    # shutil.copy(args.config, exp_path)
    os.environ['nnUNet_results'] = str(exp_path)
    # from nnunetv2.paths import nnUNet_preprocessed    

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode == 'all' and args.trainer_name == 'nnUNetTrainer':
        train_all(
            # config=config,
            dataset_ids=args.dataset_ids,
            configuration=args.configuration, 
            trainer_name=args.trainer_name, 
            fold=args.fold,
            dropout_p=args.dropout_p,
            exp_path=exp_path,
            subset=not args.all,
            from_ckpt=args.ts is not None,
        )
    if args.mode == 'all' and args.trainer_name == 'FednnUNetTrainerMarginalLoss':
        train_all_marginal(
            # config=config,
            dataset_ids=args.dataset_ids,
            configuration=args.configuration, 
            trainer_name=args.trainer_name, 
            fold=args.fold,
            dropout_p=args.dropout_p,
            exp_path=exp_path,
            subset=not args.all,
            from_ckpt=args.ts is not None,
        )
    elif args.mode == 'federated' and args.trainer_name == 'FednnUNetTrainer':
        train_federated_separate_heads(
            # config=config,
            dataset_ids=args.dataset_ids,
            configuration=args.configuration, 
            trainer_name=args.trainer_name, 
            fold=args.fold,
            dropout_p=args.dropout_p,
            fedprox_mu=args.fedprox_mu,
            exp_path=exp_path,
            from_ckpt=args.ts is not None,
            subset=not args.all
        )
    elif args.mode == 'federated' and args.trainer_name in ['ConDistFLTrainer', 'FednnUNetTrainerMarginalLoss']:
        train_federated_no_separate_heads(
            # config=config,
            dataset_ids=args.dataset_ids,
            configuration=args.configuration, 
            trainer_name=args.trainer_name, 
            fold=args.fold,
            dropout_p=args.dropout_p,
            fedprox_mu=args.fedprox_mu,
            exp_path=exp_path,
            from_ckpt=args.ts is not None,
            subset=not args.all
        )
    # else:
    #     train_single(
    #         dataset_name_or_id=datasets[0], 
    #         configuration=args.configuration, 
    #         fold=args.fold, 
    #         trainer_name=args.trainer_name,
    #         mcmc_kwargs=mcmc_config,
    #         vi_kwargs=vi_config
    #     )

    # python train.py --mode federated --exp_name ts --trainer_name FednnUNetTrainer --dataset_ids 37 --nnUNet_preprocessed '/gpfs/bwfor/work/ws/hd_cn265-fun/CT20K/ct_ready_nnunet/preprocessed'