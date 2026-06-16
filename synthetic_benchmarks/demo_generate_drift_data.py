
"""
Demo: generate synthetic data with session/domain-wise varying mixing.

Run:
  python demo_generate_drift_data.py

You should see that:
  - the same segment label u has different x statistics across domains (because mixing differs)
  - this violates "fixed mixing" assumption used by iVAE/IIA/TCL identifiability proofs.
"""

import numpy as np

from data.drift_data import generate_data_with_mixing_drift


def main():
    # --- core settings (keep small for quick sanity check) ---
    n_per_seg = 200
    n_seg = 5
    d = 3
    n_layers = 3

    # --- drift settings ---
    n_domains = 3
    drift_strength = 0.15     # increase to make drift stronger
    drift_mode = "perturb"    # "perturb" (small drift) or "independent" (new mixing each domain)

    S, X, U, M, L, extra = generate_data_with_mixing_drift(
        n_per_seg=n_per_seg,
        n_seg=n_seg,
        d_sources=d,
        d_data=d,
        n_layers=n_layers,
        prior="gauss",
        activation="lrelu",
        seed=0,
        repeat_linearity=True,          # matches the original iVAE codepath
        n_domains=n_domains,
        drift_mode=drift_mode,
        drift_strength=drift_strength,
        share_source_params=True,       # isolate effect of mixing drift
        one_hot_labels=True,
        return_extra=True,
    )

    print("S shape:", S.shape, "X shape:", X.shape, "U shape:", U.shape)
    print("z shape:", extra.z.shape, "num domains:", len(np.unique(extra.z)))

    # Check that mixing differs across domains (only available in `extra`)
    A0 = extra.mixing[0]["A"]
    A1 = extra.mixing[1]["A"]
    print("||A1 - A0||_F =", float(np.linalg.norm(A1 - A0)))

    # Check that p(x | u=0) differs across domains (it should, because mixing differs)
    z = extra.z
    u = U.argmax(axis=1)

    def stats(z_id, u_id):
        mask = (z == z_id) & (u == u_id)
        Xsub = X[mask]
        mu = Xsub.mean(axis=0)
        cov = np.cov(Xsub.T)
        return mu, cov

    mu00, cov00 = stats(0, 0)
    mu10, cov10 = stats(1, 0)

    print("||mean(z=1,u=0) - mean(z=0,u=0)|| =", float(np.linalg.norm(mu10 - mu00)))
    print("||cov(z=1,u=0)  - cov(z=0,u=0)||  =", float(np.linalg.norm(cov10 - cov00)))

    # Optional: save for later training
    # np.savez_compressed("drift_demo.npz", s=S, x=X, u=U, z=z, m=M, L=L)
    # print("Saved to drift_demo.npz")


if __name__ == "__main__":
    main()
