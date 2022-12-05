import os
import math
import torch
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .model import *
from .dataset import IEMOCAPDataset, collate_fn
from pdb import set_trace

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float = 3.0):
        super(MinimumClassConfusionLoss, self).__init__()
        if temperature == None:
            temperature = 3.0
            print(f"temperature is default value {temperature}")
        else:
            print(f"temperature is {temperature}")
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        
        
        DATA_ROOT = self.datarc['root']
        meta_data = self.datarc["meta_data"]

        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"
        
        self.mcc = MinimumClassConfusionLoss(self.datarc.get('temperature'))
        self.transfer_loss_ratio = self.datarc.get('transfer_loss_ratio')
        if self.transfer_loss_ratio == None:
            self.transfer_loss_ratio = 0.1
        print(f"transfer loss ratio={self.transfer_loss_ratio}")
        
        print(f"[Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")

        train_path = os.path.join(
            meta_data, self.fold.replace('fold', 'Session'), 'train_meta_data.json')
        print(f'[Expert] - Training path: {train_path}')

        test_path = os.path.join(
            meta_data, self.fold.replace('fold', 'Session'), 'test_meta_data.json')
        print(f'[Expert] - Testing path: {test_path}')
        
        dataset = IEMOCAPDataset(DATA_ROOT, train_path, self.datarc['pre_load'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        
        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = IEMOCAPDataset(DATA_ROOT, test_path, self.datarc['pre_load'])
        # self.test_dataset = dataset

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = dataset.class_num,
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))


    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, is_train = True, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        #set_trace()
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        cls_loss, mcc_loss = 0, 0
        if is_train:
            cls_loss = self.objective(predicted, labels)
        else:
            mcc_loss = self.mcc(predicted)
        loss = cls_loss + self.transfer_loss_ratio * mcc_loss

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)

        return save_names
