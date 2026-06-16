import argparse
import os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='demo_data/demo_cross_domain_eeg.npz')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-per-domain', type=int, default=240)
    p.add_argument('--n-features', type=int, default=20)
    p.add_argument('--n-classes', type=int, default=4)
    p.add_argument('--n-periods', type=int, default=12)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    Xs, ys, ds, us = [], [], [], []
    class_proto = rng.normal(size=(args.n_classes, args.n_features)).astype(np.float32)
    period_shift = rng.normal(scale=0.25, size=(args.n_periods, args.n_features)).astype(np.float32)
    for d, dom in enumerate(['ref', 'target']):
        A = np.eye(args.n_features) + (0.12 * d) * rng.normal(size=(args.n_features, args.n_features))
        for i in range(args.n_per_domain):
            y = i % args.n_classes
            u = int(np.floor(i * args.n_periods / args.n_per_domain))
            s = class_proto[y] + period_shift[u] + rng.normal(scale=0.5, size=args.n_features)
            x = s @ A + rng.normal(scale=0.1, size=args.n_features)
            Xs.append(x.astype(np.float32)); ys.append(y); ds.append(dom); us.append(u)
    X = np.stack(Xs)
    y = np.array(ys)
    domain = np.array(ds)
    u = np.array(us)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, domain=domain, u=u)
    print(f'Saved demo dataset: {args.out}')


if __name__ == '__main__':
    main()
