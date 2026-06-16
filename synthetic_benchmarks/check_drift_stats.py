import yaml, numpy as np
from data.drift_data import generate_data_with_mixing_drift

cfg = yaml.safe_load(open("configs/ivae_drift.yaml", "r"))

S, X, U, M, L, extra = generate_data_with_mixing_drift(
    n_per_seg=cfg["nps"],
    n_seg=cfg["ns"],
    d_sources=cfg["dl"],
    d_data=cfg["dd"],
    n_layers=cfg["nl"],
    prior=cfg["p"],
    activation=cfg["act"],
    seed=cfg["s"],
    repeat_linearity=True,
    one_hot_labels=True,
    n_domains=cfg["n_domains"],
    drift_mode=cfg["drift_mode"],
    drift_strength=cfg["drift_strength"],
    share_source_params=cfg["share_source_params"],
    return_extra=True,
)

z = extra.z
u = U.argmax(axis=1)

m0 = extra.mixing[0]
for dom in range(1, cfg["n_domains"]):
    md = extra.mixing[dom]
    dA = np.linalg.norm(md["A"] - m0["A"])
    dB = np.linalg.norm(md["B"] - m0["B"])
    print(f"domain {dom}: ||A-A0||={dA:.4f}, ||B-B0||={dB:.4f}")

# same u=0, compare stats across domain 0 and 1
def stats(dom_id, u_id):
    mask = (z == dom_id) & (u == u_id)
    Xsub = X[mask]
    mu = Xsub.mean(axis=0)
    cov = np.cov(Xsub.T)
    return mu, cov

mu00, cov00 = stats(0, 0)
mu10, cov10 = stats(1, 0)
print("||mean(z=1,u=0)-mean(z=0,u=0)|| =", np.linalg.norm(mu10-mu00))
print("||cov(z=1,u=0)-cov(z=0,u=0)||  =", np.linalg.norm(cov10-cov00))
