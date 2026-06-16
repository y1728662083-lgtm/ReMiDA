import argparse
import os
import pickle
import sys

import numpy as np
import torch
import yaml

from runners import ivae_runner, tcl_runner, eeg_runner


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str, default='ivae.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')
    parser.add_argument('--data', type=str, default='', help='Optional dataset path override (e.g., ds003626 .npz)')

    parser.add_argument('--n-sims', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def _safe_doc_name(doc: str, config_path: str = ''):
    if doc is None:
        doc = ''
    doc = str(doc).strip()
    if doc == '':
        base = os.path.splitext(os.path.basename(str(config_path)))[0]
        doc = base if base else 'run'
    doc = doc.replace('\\', '__').replace('/', '__')
    return doc


def make_dirs(args):
    args.doc = _safe_doc_name(getattr(args, 'doc', ''), getattr(args, 'config', ''))
    os.makedirs(args.run, exist_ok=True)
    args.log = os.path.join(args.run, 'logs', args.doc)
    os.makedirs(args.log, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.doc)
    os.makedirs(args.checkpoints, exist_ok=True)
    args.data_path = os.path.join(args.run, 'datasets', args.doc)
    os.makedirs(args.data_path, exist_ok=True)


def main():
    args = parse()
    make_dirs(args)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if new_config.tcl:
        r = tcl_runner(args, new_config)
    else:
        dataset = str(getattr(new_config, 'dataset', 'synthetic')).lower()
        if dataset in {'ds003626_npz', 'eeg', 'eeg_npz'} or bool(getattr(new_config, 'real_eeg', False)):
            r = eeg_runner(args, new_config)
        else:
            r = ivae_runner(args, new_config)
    safe_cfg_stem = _safe_doc_name(os.path.splitext(args.config)[0], args.config)
    fname = os.path.join(args.run, '_'.join([safe_cfg_stem, str(args.seed), str(args.n_sims)]) + '.p')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    pickle.dump(r, open(fname, "wb"))


if __name__ == '__main__':
    sys.exit(main())
