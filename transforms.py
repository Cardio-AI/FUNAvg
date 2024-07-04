import torch

class LabelTransform:
    def __init__(self, all_labels_in_dataset, wanted_labels_in_dataset):
        self.all_labels_in_dataset = {int(k): v for k, v in all_labels_in_dataset.items()}
        self.wanted_labels_in_dataset = wanted_labels_in_dataset
    
    def __call__(self, data):
        label = data['target']
        if type(label) == list:
            new_label = [torch.zeros_like(l) for l in label]
        else:
            new_label = [torch.zeros_like(label)]
            label = [label]
        for l, nl in zip(label, new_label):
            for i in range(1, len(self.all_labels_in_dataset)):
                al = self.all_labels_in_dataset[i]
                if al in self.wanted_labels_in_dataset:
                    nl[l == i] = self.wanted_labels_in_dataset[al]
        data['target'] = [l.float() for l in new_label]
        data['cur_task'] = list(self.wanted_labels_in_dataset.keys())
        return data

class LabelTransformNNUnet(LabelTransform):
    def __call__(self, data, target, keys, properties):
        _data = super().__call__({'target': target})
        _data['data'] = data.float()
        return _data