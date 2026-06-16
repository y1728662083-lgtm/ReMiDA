import os
import copy
import argparse
import yaml

def slug(x: float) -> str:
    # 0.05 -> 0p05
    s = f"{x}".replace(".", "p")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="base yaml file name under configs/, e.g. ivae_drift_pooled_independent.yaml")
    ap.add_argument("--prefix", type=str, required=True, help="output prefix, e.g. sweep_pooled_indep")
    ap.add_argument("--mode", type=str, default="perturb", choices=["perturb", "independent"], help="drift_mode")
    ap.add_argument("--strengths", type=float, nargs="+", required=True, help="list of drift_strength values")
    args = ap.parse_args()

    base_path = os.path.join("configs", args.base)
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_names = []
    for ds in args.strengths:
        c = copy.deepcopy(cfg)
        c["mix_drift"] = True
        c["drift_mode"] = args.mode
        c["drift_strength"] = float(ds)

        out_name = f"{args.prefix}_ds{slug(ds)}.yaml"
        out_path = os.path.join("configs", out_name)
        with open(out_path, "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)

        out_names.append(out_name)

    print("Generated YAMLs:")
    for n in out_names:
        print("  ", n)

if __name__ == "__main__":
    main()