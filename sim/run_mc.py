import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time, pandas as pd
from multiprocessing import Pool
from ph2sim import DISTS, one_rep
R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N, P = 2000, 0.9995
KGRID = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1000]
COLS = "dist seed k aH qH aG qG aA qA rhoA nllA aB qB nllB".split()
if __name__ == "__main__":
    jobs = [(nm, 1000 * i + j, N, KGRID, P) for i, nm in enumerate(DISTS) for j in range(R)]
    rows, t0 = [], time.time()
    with Pool(4) as pool:
        for i, out in enumerate(pool.imap_unordered(one_rep, jobs, chunksize=2), 1):
            rows.extend(out)
            if i % 100 == 0:
                el = time.time() - t0
                print(f"{i}/{len(jobs)} reps  {el/60:.1f} min  eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
                pd.DataFrame(rows, columns=COLS).to_csv("mc_results.csv", index=False)
    pd.DataFrame(rows, columns=COLS).to_csv("mc_results.csv", index=False)
    print("done", len(rows), flush=True)
