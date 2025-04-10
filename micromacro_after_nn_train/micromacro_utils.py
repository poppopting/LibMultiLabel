import os
import sys
import glob
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

sys.path.append('./../')
from main import get_config
from torch_trainer import TorchTrainer
from libmultilabel.nn.metrics import get_metrics

def load_config(method, data_name, data_type='retrain', model_name='best_model.ckpt', runs_base_dir='./runs'):
    runs_dir = glob.glob(os.path.join(runs_base_dir, f'{method}-{data_name}-{data_type}*'))[0]
    
    log_path = os.path.join(runs_dir, 'logs.json')
    log = json.load(open(log_path, 'r'))
    config_path = log['config']['config']

    sys.argv = ['']
    sys.argv.extend(['--config', config_path])
    config = get_config()

    trainer = TorchTrainer(config)

    os.system(f'rm -r {config["checkpoint_dir"]} || true')
    checkpoint_dir = log['config']['checkpoint_dir']
    model_path = os.path.join(checkpoint_dir, model_name)
    trainer._setup_model(checkpoint_path=model_path, log_path=log_path)

    # trainer.model.eval()
    # if 'val' in trainer.datasets:
    #     val_loader = trainer._get_dataset_loader('val', shuffle=False)
    # else: 
    #     print('no val data')
    #     val_loader = None
    # test_loader = trainer._get_dataset_loader('test', shuffle=False)

    return trainer

def get_predict_value(trainer, split):

    predict_value = defaultdict(list)

    trainer.model.eval()   
    device = trainer.model.device
    
    if split not in trainer.datasets:
        print(f'no {split} in trainer.datasets')
        return {'label': torch.tensor([]), 'logits': torch.tensor([])}

    loader = trainer._get_dataset_loader(split, shuffle=False)

    for batch in tqdm(loader):
        batch["text"] = batch["text"].to(device)
        predict_value['label'].append(batch['label'].detach().cpu()) # .numpy()
        predict_value['logits'].append(trainer.model(batch).detach().cpu()) # .numpy()

    predict_value['label'] = torch.cat(predict_value['label'], dim=0)
    predict_value['logits'] = torch.cat(predict_value['logits'], dim=0)

    torch.cuda.empty_cache() 
    return predict_value

def dump_predict_value(method, data_name, data_type='retrain', splits=['val', 'test'], model_name='best_model.ckpt', runs_base_dir='./runs', save_folder='./predict_values'):

    save_dir = f'{save_folder}/{data_name}/{method}'
    os.makedirs(save_dir, exist_ok=True)
 
    trainer = load_config(method, data_name, data_type, model_name, runs_base_dir)
    
    for split in splits:
        predict_value = get_predict_value(trainer, split)
        logits, label = predict_value['logits'].numpy(), predict_value['label'].numpy()
        prefix = 'fold_' if data_type != 'retrain' else ''
        file_name = f'{prefix}{data_type}_{split}_predict_values.npz'
        np.savez_compressed(os.path.join(save_dir, file_name), logits=logits, label=label)
    return

def evaluate_micromaro(preds, target, naive=True, logits_based_thresholds=None, simgoid_based_thresholds=None):
    metrics = defaultdict(list)

    preds, target = torch.from_numpy(preds), torch.from_numpy(target)

    # before micromacro
    if naive:
        metric = get_metrics(0., ['Micro-F1', 'Macro-F1'], num_classes=target.shape[1], top_k=None)
        metric.update(preds, target)
        metrics['naive'].append(metric.compute())

    if logits_based_thresholds is not None:
        # after micromacro using raw logits
        metric = get_metrics(torch.from_numpy(logits_based_thresholds), ['Micro-F1', 'Macro-F1'], num_classes=target.shape[1], top_k=None)
        metric.update(preds, target)
        metrics['logits'].append(metric.compute())

    if simgoid_based_thresholds is not None:
        # after micromacro using simgoid logits
        metric = get_metrics(torch.from_numpy(simgoid_based_thresholds), ['Micro-F1', 'Macro-F1'], num_classes=target.shape[1], top_k=None)
        metric.update(torch.sigmoid(preds), target)
        metrics['sigmoid'].append(metric.compute())

    return metrics