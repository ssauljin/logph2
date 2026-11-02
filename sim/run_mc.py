import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time, pandas as pd
from multiprocessing import Pool
from ph2sim import DISTS, one_rep
# usage: python3 run_mc.py R N OUTFILE      (VaR level p = 1 - 1/N, k grid scales with N)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OUT = sys.argv[3] if len(sys.argv) > 3 else "mc_results.csv"
P = 1 - 1 / N
KGRID = [int(round(f * N)) for f in (0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)]
L = "aH qH aA qA rhoA nllA aB qB nllB".split()
COLS = ["dist", "seed", "k", "aG", "qG"] + L + [c + "_s" for c in L]
if __name__ == "__main__":
    jobs = [(nm, 1000 * i + j, N, KGRID, P) for i, nm in enumerate(DISTS) for j in range(R)]
    rows, t0 = [], time.time()
    with Pool(4) as pool:
        for i, out in enumerate(pool.imap_unordered(one_rep, jobs, chunksize=2), 1):
            rows.extend(out)
            if i % 100 == 0:
                el = time.time() - t0
                print(f"{i}/{len(jobs)} reps  {el/60:.1f} min  eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
                pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)
    pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)
    print("done", len(rows), flush=True)
