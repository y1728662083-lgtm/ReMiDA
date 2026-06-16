import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.common import (
    ensure_dir, fit_standardizer, load_npz_dataset, load_split, make_reference_target_split,
    normalize_u, save_features, save_split, set_seed, train_downstream_for_input_modes,
    encode_1d_labels,
)


class TCLEncoder(nn.Module):
    """TCL-style period-discrimination encoder implemented in PyTorch for a TF1-free baseline.

    It follows the TCL idea used in the uploaded baseline: learn a representation by discriminating
    non-stationary shared-period labels. It is provided because the original TCL code requires TF1.x.
    This cleaned package uses a PyTorch TCL-style period-prediction baseline to avoid the TensorFlow-1.x dependency of the original TCL code.
    """
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, n_periods: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.head = nn.Linear(latent_dim, n_periods)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return z, logits


def parse_args():
    p = argparse.ArgumentParser(description='Paper-aligned PyTorch TCL baseline for ReMiDA-CLSRNet.')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ref-domain', required=True)
    p.add_argument('--target-domain', required=True)
    p.add_argument('--split-file', default=None)
    p.add_argument('--n-periods', type=int, default=20)
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default=None)
    p.add_argument('--pca-dim', type=int, default=256, help='PCA dimension fitted only on reftrain+targetadapt.')
    p.add_argument('--input-clip', type=float, default=8.0)
    p.add_argument('--downstream-inputs', nargs='+', default=['all'], choices=['all','latent','raw_pca','raw_plus_latent'])
    p.add_argument('--clf-epochs', type=int, default=120)
    p.add_argument('--clf-lr', type=float, default=1e-3)
    p.add_argument('--clf-hidden-dim', type=int, default=128)
    p.add_argument('--clf-patience', type=int, default=25)
    return p.parse_args()


def sanitize_features(X, name='X'):
    X = np.asarray(X, dtype=np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        print(f'[preprocess] {name}: replacing non-finite values: n={int(bad.sum())}', flush=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X


def fit_pca_transform(X_std, train_idx, pca_dim, seed):
    if pca_dim is None or int(pca_dim) <= 0 or int(pca_dim) >= X_std.shape[1]:
        return X_std.astype(np.float32), None, float('nan')
    n_components = min(int(pca_dim), len(train_idx) - 1, X_std.shape[1])
    print(f'[preprocess] fitting PCA on reftrain+targetadapt only: input_dim={X_std.shape[1]}, pca_dim={n_components}', flush=True)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
    pca.fit(X_std[train_idx])
    Xp = pca.transform(X_std).astype(np.float32)
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f'[preprocess] PCA explained_variance_ratio_sum={explained:.4f}', flush=True)
    return Xp, pca, explained


def train_tcl_encoder(X_train, u_train, args, device, n_periods):
    model = TCLEncoder(X_train.shape[1], args.latent_dim, args.hidden_dim, n_periods).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(u_train).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for xb, ub in loader:
            xb, ub = xb.to(device), ub.to(device)
            opt.zero_grad(set_to_none=True)
            _, logits = model(xb)
            loss = loss_fn(logits, ub)
            if not torch.isfinite(loss):
                raise FloatingPointError('TCL loss became NaN/Inf. Try lower --lr or --pca-dim.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            correct += int((logits.argmax(dim=1) == ub).sum().item())
            seen += len(xb)
        row = {'epoch': ep, 'loss': total / max(seen, 1), 'period_train_acc': correct / max(seen, 1)}
        history.append(row)
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[TCL-pytorch] epoch {ep:03d}/{args.epochs}, loss={row['loss']:.6f}, period_acc={row['period_train_acc']:.4f}", flush=True)
    return model, history


@torch.no_grad()
def extract_tcl_features(model, X, device, batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size, shuffle=False)
    outs = []
    for (xb,) in loader:
        xb = xb.to(device)
        z, _ = model(xb)
        outs.append(z.detach().cpu().numpy())
    return sanitize_features(np.concatenate(outs, axis=0).astype(np.float32), 'Z_tcl')


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[setup] device={device}', flush=True)

    data = load_npz_dataset(args.data)
    X = sanitize_features(data['X'], 'X')
    y, y_mapping = encode_1d_labels(data['y'])
    domains = data['domain'].astype(str)
    u_label, _ = normalize_u(data.get('u', None), domains, args.n_periods)
    n_periods = int(np.max(u_label)) + 1

    if args.split_file:
        split = load_split(args.split_file)
        print(f'[split] loaded split from {args.split_file}', flush=True)
    else:
        split = make_reference_target_split(y, domains, args.ref_domain, args.target_domain, seed=args.seed)
        save_split(os.path.join(args.out, 'split_indices.npz'), split)
    rep_train_idx = np.concatenate([split.reftrain, split.targetadapt])
    print(f'[split] {args.ref_domain}->{args.target_domain}: reftrain={len(split.reftrain)}, refval={len(split.refval)}, targetadapt={len(split.targetadapt)}, targettest={len(split.targettest)}', flush=True)

    X_std, _ = fit_standardizer(X, rep_train_idx)
    if args.input_clip and args.input_clip > 0:
        X_std = np.clip(X_std, -float(args.input_clip), float(args.input_clip)).astype(np.float32)
    X_model, pca, explained = fit_pca_transform(X_std, rep_train_idx, args.pca_dim, args.seed)
    if args.input_clip and args.input_clip > 0:
        X_model = np.clip(X_model, -float(args.input_clip), float(args.input_clip)).astype(np.float32)
    X_model = sanitize_features(X_model, 'X_model')

    model, history = train_tcl_encoder(X_model[rep_train_idx], u_label[rep_train_idx], args, device, n_periods)
    Z = extract_tcl_features(model, X_model, device, batch_size=max(args.batch_size, 256))

    feature_bank = {
        'latent': Z,
        'raw_pca': X_model,
        'raw_plus_latent': np.concatenate([X_model, Z], axis=1).astype(np.float32),
    }

    torch.save(model.state_dict(), os.path.join(args.out, 'tcl_pytorch_model.pt'))
    if pca is not None:
        with open(os.path.join(args.out, 'pca.pkl'), 'wb') as f:
            pickle.dump(pca, f)
    with open(os.path.join(args.out, 'tcl_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(y_mapping, f, indent=2, ensure_ascii=False)
    meta = {
        'method': 'TCL_pytorch_period_discrimination',
        'original_uploaded_TCL_code_preserved': True,
        'original_TCL_path': 'not included in clean package; PyTorch TCL-style baseline used',
        'why_pytorch': 'The uploaded TCL baseline is TensorFlow 1.x code. This script implements the same TCL period-discrimination baseline in PyTorch for the current AutoDL environment.',
        'ref_domain': args.ref_domain,
        'target_domain': args.target_domain,
        'split': {k: int(len(v)) for k, v in split.as_dict().items()},
        'raw_flat_dim': int(X.shape[1]),
        'model_input_dim': int(X_model.shape[1]),
        'pca_dim': int(args.pca_dim),
        'pca_explained_variance_ratio_sum': explained,
        'latent_dim': int(args.latent_dim),
        'note_on_inputs': {
            'latent': 'TCL encoder representation learned by shared-period discrimination',
            'raw_pca': 'same raw EEG trials after standardization+PCA, vector MLP classifier',
            'raw_plus_latent': 'concat(raw_pca, Z_TCL)',
            'a1': 'not produced by TCL because a1 is ReMiDA-specific aligned observation'
        }
    }
    with open(os.path.join(args.out, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    save_features(os.path.join(args.out, 'features_tcl.npz'), Z, y, u_label, domains, split, extra={'X_model_raw_pca': X_model.astype(np.float32)})
    metrics = train_downstream_for_input_modes(
        feature_bank, y, split, args.out, input_modes=args.downstream_inputs,
        seed=args.seed, epochs=args.clf_epochs, batch_size=args.batch_size,
        lr=args.clf_lr, hidden_dim=args.clf_hidden_dim, patience=args.clf_patience,
        device=str(device)
    )
    print('[TCL-pytorch] metrics_by_input:', json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
