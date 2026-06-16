import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
# Reuse the cleaned synthetic iVAE implementation from the package root.
PACKAGE_ROOT = os.path.abspath(os.path.join(ROOT, '..'))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, 'synthetic_benchmarks'))

# Reused iVAE implementation.
from models.nets import iVAE
from src.common import (
    ensure_dir,
    fit_standardizer,
    load_npz_dataset,
    load_split,
    make_reference_target_split,
    normalize_u,
    save_features,
    save_split,
    set_seed,
    train_downstream_for_input_modes,
    encode_1d_labels,
)


def parse_args():
    p = argparse.ArgumentParser(description='Paper-aligned stable iVAE baseline for ReMiDA-CLSRNet.')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ref-domain', required=True)
    p.add_argument('--target-domain', required=True)
    p.add_argument('--split-file', default=None, help='Reuse an existing reftrain/refval/targetadapt/targettest split.')
    p.add_argument('--n-periods', type=int, default=20)
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--activation', default='lrelu', choices=['lrelu', 'xtanh', 'sigmoid', 'none'])
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default=None)
    p.add_argument('--pca-dim', type=int, default=256, help='PCA dimension fitted only on reftrain+targetadapt. Set 0 to disable.')
    p.add_argument('--input-clip', type=float, default=8.0)
    p.add_argument('--decoder-var', type=float, default=1.0)
    p.add_argument('--clamp-logvar', type=float, default=6.0)
    p.add_argument('--downstream-inputs', nargs='+', default=['all'], choices=['all','latent','raw_pca','raw_plus_latent'],
                   help='Which feature(s) are fed to the downstream MLP. all = latent, raw_pca, raw_plus_latent.')
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


def enable_stable_variances(model, clamp):
    if clamp is None or float(clamp) <= 0:
        return
    clamp = float(clamp)

    def encoder_params_clamped(x, u):
        xu = torch.cat((x, u), 1)
        g = model.g(xu)
        logv = torch.clamp(model.logv(xu), -clamp, clamp)
        return g, logv.exp()

    def prior_params_clamped(u):
        logl = torch.clamp(model.logl(u), -clamp, clamp)
        return model.prior_mean, logl.exp()

    model.encoder_params = encoder_params_clamped
    model.prior_params = prior_params_clamped


def train_ivae_model(X_train, U_train, args, device):
    model = iVAE(
        latent_dim=args.latent_dim,
        data_dim=X_train.shape[1],
        aux_dim=U_train.shape[1],
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        device=device,
        anneal=False,
    ).to(device)
    model.decoder_var = torch.ones(1, device=device) * float(args.decoder_var)
    enable_stable_variances(model, args.clamp_logvar)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(U_train).float()), batch_size=args.batch_size, shuffle=True, drop_last=False)
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, ub in loader:
            xb, ub = xb.to(device), ub.to(device)
            opt.zero_grad(set_to_none=True)
            elbo, _ = model.elbo(xb, ub)
            loss = -elbo
            if not torch.isfinite(loss):
                raise FloatingPointError('iVAE loss became NaN/Inf. Try smaller --lr, lower --pca-dim, larger --decoder-var, or lower --clamp-logvar.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            n_seen += len(xb)
        avg = total / max(n_seen, 1)
        history.append({'epoch': ep, 'loss': avg})
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f'[iVAE] epoch {ep:03d}/{args.epochs}, loss={avg:.6f}', flush=True)
    return model, history


@torch.no_grad()
def extract_encoder_mean(model, X, U, device, batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(U).float()), batch_size=batch_size, shuffle=False)
    outs = []
    for xb, ub in loader:
        xb, ub = xb.to(device), ub.to(device)
        g, _ = model.encoder_params(xb, ub)
        outs.append(g.detach().cpu().numpy())
    return sanitize_features(np.concatenate(outs, axis=0).astype(np.float32), 'Z_ivae')


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
    u_label, U = normalize_u(data.get('u', None), domains, args.n_periods)

    if args.split_file:
        split = load_split(args.split_file)
        print(f'[split] loaded split from {args.split_file}', flush=True)
    else:
        split = make_reference_target_split(y, domains, args.ref_domain, args.target_domain, seed=args.seed)
        save_split(os.path.join(args.out, 'split_indices.npz'), split)
    rep_train_idx = np.concatenate([split.reftrain, split.targetadapt])
    print(f'[split] {args.ref_domain}->{args.target_domain}: reftrain={len(split.reftrain)}, refval={len(split.refval)}, targetadapt={len(split.targetadapt)}, targettest={len(split.targettest)}', flush=True)

    X_std, _ = fit_standardizer(X, rep_train_idx)
    X_std = sanitize_features(X_std, 'X_std')
    if args.input_clip and args.input_clip > 0:
        X_std = np.clip(X_std, -float(args.input_clip), float(args.input_clip)).astype(np.float32)
    X_model, pca, explained = fit_pca_transform(X_std, rep_train_idx, args.pca_dim, args.seed)
    if args.input_clip and args.input_clip > 0:
        X_model = np.clip(X_model, -float(args.input_clip), float(args.input_clip)).astype(np.float32)
    X_model = sanitize_features(X_model, 'X_model')

    model, history = train_ivae_model(X_model[rep_train_idx], U[rep_train_idx], args, device)
    Z = extract_encoder_mean(model, X_model, U, device, batch_size=max(args.batch_size, 256))

    feature_bank = {
        'latent': Z,
        'raw_pca': X_model,
        'raw_plus_latent': np.concatenate([X_model, Z], axis=1).astype(np.float32),
    }

    torch.save(model.state_dict(), os.path.join(args.out, 'ivae_model.pt'))
    if pca is not None:
        with open(os.path.join(args.out, 'pca.pkl'), 'wb') as f:
            pickle.dump(pca, f)
    with open(os.path.join(args.out, 'ivae_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(y_mapping, f, indent=2, ensure_ascii=False)
    meta = {
        'method': 'iVAE_stable_reimplemented',
        'uses_uploaded_original_ivae_code': True,
        'ref_domain': args.ref_domain,
        'target_domain': args.target_domain,
        'split': {k: int(len(v)) for k, v in split.as_dict().items()},
        'raw_flat_dim': int(X.shape[1]),
        'model_input_dim': int(X_model.shape[1]),
        'pca_dim': int(args.pca_dim),
        'pca_explained_variance_ratio_sum': explained,
        'latent_dim': int(args.latent_dim),
        'note_on_inputs': {
            'latent': 'standard iVAE encoder mean Z_iVAE; comparable to latent-only baseline, not equal to proposed ReMiDA S-iVAE l1',
            'raw_pca': 'same raw EEG trials after standardization+PCA, vector MLP classifier; not the paper EEGNet r1',
            'raw_plus_latent': 'concat(raw_pca, Z_iVAE); analogous to raw+latent idea but still vector MLP, not proposed rL',
            'a1': 'not produced by iVAE/TCL baselines because a1 is ReMiDA-specific aligned observation'
        }
    }
    with open(os.path.join(args.out, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    save_features(os.path.join(args.out, 'features_ivae.npz'), Z, y, u_label, domains, split, extra={'X_model_raw_pca': X_model.astype(np.float32)})
    metrics = train_downstream_for_input_modes(
        feature_bank, y, split, args.out, input_modes=args.downstream_inputs,
        seed=args.seed, epochs=args.clf_epochs, batch_size=args.batch_size,
        lr=args.clf_lr, hidden_dim=args.clf_hidden_dim, patience=args.clf_patience,
        device=str(device)
    )
    print('[iVAE] metrics_by_input:', json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
