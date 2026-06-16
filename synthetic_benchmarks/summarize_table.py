import re, numpy as np, sys, pathlib

def parse(path):
    txt = pathlib.Path(path).read_text(errors="ignore")

    full = [float(x) for x in re.findall(r"Final MCC on FULL dataset \(seed=\d+\):\s*([0-9.]+)", txt)]
    cons = re.findall(
        r"\[diag\]\s*identity consistency vs ref\(z=.*?\):\s*perm_agreement=([0-9.]+),\s*sign_agreement=([0-9.]+)",
        txt
    )
    permd = [float(a) for a, b in cons]
    signd = [float(b) for a, b in cons]
    probe = [float(x) for x in re.findall(r"session leakage probe .*acc=([0-9.]+)", txt)]

    def ms(a):
        a = np.array(a, dtype=float)
        if a.size == 0:
            return (np.nan, np.nan)
        return (float(a.mean()), float(a.std(ddof=0)))

    return {
        "FULL_MCC_mean,std": ms(full),
        "perm_agree_mean,std": ms(permd),
        "sign_agree_mean,std": ms(signd),
        "probe_acc_mean,std": ms(probe),
        "n_seeds": len(full),
    }

for p in sys.argv[1:]:
    r = parse(p)
    print("\n===", p, "===")
    for k,v in r.items():
        print(k, v)