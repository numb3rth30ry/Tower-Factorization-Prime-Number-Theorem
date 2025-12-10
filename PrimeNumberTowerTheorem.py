#!/usr/bin/env python3
"""
mu2_large_diagnostics.py

Compute mu_2(n) for n <= N efficiently and run diagnostics probing PNT-like cancellation.

Usage:
    python mu2_large_diagnostics.py --N 5000000 --out mu2_5e6.npz --plot

Outputs to stdout and (optionally) plots. Saves arrays to the given --out .npz file.

Author: delivered for your local execution — paste & run.
"""
import sys
import argparse
import math
import time
from array import array
import numpy as np

def sieve_spf(n):
    """Return smallest-prime-factor list of length n+1 (spf[0]=0, spf[1]=1)."""
    spf = list(range(n+1))
    if n >= 0: spf[0] = 0
    if n >= 1: spf[1] = 1
    lim = int(n**0.5) + 1
    for p in range(2, lim):
        if spf[p] == p:
            step = p
            start = p*p
            for k in range(start, n+1, step):
                if spf[k] == k:
                    spf[k] = p
    return spf

def is_squarefree_small(m, spf_small):
    """Return True iff m is squarefree (m small). Uses spf_small up to m."""
    if m <= 1:
        return True
    tmp = m
    while tmp > 1:
        p = spf_small[tmp]
        cnt = 0
        while tmp % p == 0:
            tmp //= p
            cnt += 1
            if cnt >= 2:
                return False
    return True

def omega_small(m, spf_small):
    """Number of distinct prime divisors of small m (uses spf_small)."""
    if m <= 1:
        return 0
    tmp = m
    count = 0
    while tmp > 1:
        p = spf_small[tmp]
        count += 1
        while tmp % p == 0:
            tmp //= p
    return count

def compute_mu2_up_to(N, show_progress=True):
    """Compute mu2[1..N] and partial sums S[1..N]. Returns (mu2_array (int8), S_array (int64))."""
    t0 = time.time()
    spf = sieve_spf(N)
    # small SPF up to exponent max
    emax = max(64, int(math.log2(max(2, N))) + 8)
    spf_small = sieve_spf(emax)

    # mu2 stored as signed byte (-1,0,1)
    mu2 = array('b', [0]) * (N+1)
    # partial sums as 64-bit ints
    S = np.zeros(N+1, dtype=np.int64)

    running = 0
    last_report = time.time()
    report_every = max(1, N // 20)

    for n in range(1, N+1):
        # factorize n using spf
        tmp = n
        total_labels = 0
        bad = False
        while tmp > 1:
            p = spf[tmp]
            exp = 0
            while tmp % p == 0:
                tmp //= p
                exp += 1
            # check exponent exp is squarefree
            if not is_squarefree_small(exp, spf_small):
                bad = True
                break
            w = omega_small(exp, spf_small) if exp > 1 else 0
            total_labels += 1 + w
        if bad:
            val = 0
        else:
            val = -1 if (total_labels % 2 == 1) else 1
        mu2[n] = val
        running += val
        S[n] = running

        # progress
        if show_progress and (n % report_every == 0 or n == N):
            now = time.time()
            pct = n / N * 100.0
            rate = n / (now - t0 + 1e-9)
            est = (N - n) / max(rate,1e-9)
            print(f"[{time.strftime('%H:%M:%S')}] n={n:,} ({pct:.1f}%), rate={rate:.0f}/s, ETA={est:.0f}s")
            sys.stdout.flush()

    t1 = time.time()
    print(f"Compute finished in {t1-t0:.2f}s")
    return mu2, S

# ---------------- Diagnostics -----------------
def dyadic_points_up_to(N):
    """Return sorted list of dyadic points 2^k ≤ N (plus N itself)."""
    pts = []
    k = 0
    while (1 << k) <= N:
        pts.append(1 << k)
        k += 1
    if pts[-1] != N:
        pts.append(N)
    return pts

def loglog_regression(xs, ys):
    """Fit log|y| = A + beta log x using least squares on positive |y| only."""
    data = [(math.log(x), math.log(abs(y))) for x,y in zip(xs,ys) if y != 0 and x>0]
    if len(data) < 3:
        return None, None
    X = [d[0] for d in data]
    Y = [d[1] for d in data]
    xm = sum(X)/len(X); ym = sum(Y)/len(Y)
    num = sum((xi-xm)*(yi-ym) for xi,yi in zip(X,Y))
    den = sum((xi-xm)**2 for xi in X)
    beta = num / den
    A = ym - beta * xm
    C = math.exp(A)
    return C, beta

def compute_autocorrelation(mu2_arr, max_lag=2000):
    """Compute autocorrelation for lags 1..max_lag of the sequence mu2[1..N]."""
    N = len(mu2_arr) - 1
    # convert to numpy int8 for speed
    seq = np.frombuffer(mu2_arr, dtype=np.int8)[1:].astype(np.int32)  # length N
    mean = seq.mean()
    var = seq.var(ddof=0)
    if var == 0:
        return [0.0]*max_lag
    ac = []
    # compute via FFT would be faster for huge N, but we only want small lags, so direct sums are OK
    for lag in range(1, max_lag+1):
        cov = (seq[:-lag] - mean) @ (seq[lag:] - mean) / (N - lag)
        ac.append(cov / var)
    return ac

def run_diagnostics(mu2, S, out_prefix, do_plots=False):
    N = len(mu2) - 1
    nonzero = sum(1 for i in range(1,N+1) if mu2[i] != 0)
    sN = int(S[N])
    print("\n=== Basic summary ===")
    print(f"N = {N:,}")
    print(f"Nonzero mu2 count = {nonzero:,}, proportion = {nonzero/N:.6f}")
    print(f"S(N) = {sN}, S(N)/N = {sN/N:.6e}, S(N)/sqrt(N) = {sN/math.sqrt(N):.6e}")

    # Dyadic diagnostics
    dy = dyadic_points_up_to(N)
    dy_S = [int(S[x]) for x in dy]
    print("\nDyadic samples (n, S(n), S(n)/n, S(n)/sqrt(n)):")
    for x, s in zip(dy, dy_S):
        print(f"{x:12d} {s:8d} {s/x:11.4e} {s/math.sqrt(x):11.4e}")

    # log-log regression on dyadic points (skip tiny ones)
    xs = dy[3:] if len(dy) > 6 else dy
    ys = [int(S[x]) for x in xs]
    C, beta = loglog_regression(xs, ys)
    if beta is not None:
        print(f"\nLog-log fit on dyadics (|S(x)| ~ C x^beta): beta = {beta:.5f}, C = {C:.5g}")
    else:
        print("\nLog-log fit not available (insufficient nonzero points)")

    # variance growth: compute mean of S(x)^2 / x over dyadic points
    ratios = [ (S[x]**2) / x for x in xs if x>0 ]
    mean_ratio = float(sum(ratios) / len(ratios)) if ratios else float('nan')
    print(f"Mean of S(x)^2/x over dyadics = {mean_ratio:.6e}")
    # also print last few ratio values
    print("Last dyadic S(x)^2/x values (last 6):")
    for x in xs[-6:]:
        val = (S[x]**2) / x
        print(f"  n={x:12d}  S^2/n = {val:.6e}")

    # autocorrelation test
    max_lag = 500
    print("\nComputing autocorrelations up to lag", max_lag, " (this may take a moment)...")
    ac = compute_autocorrelation(mu2, max_lag=max_lag)
    # summarize: max absolute autocorrelation, first few values
    absmax = max(abs(a) for a in ac)
    print(f"Max abs autocorrelation (lags 1..{max_lag}) = {absmax:.6e}")
    print("Autocorr at small lags (1..10):")
    for i in range(10):
        print(f"lag {i+1:3d}: {ac[i]: .6e}")

    # histogram of normalized partial sums S(x)/sqrt(x) at dyadic points
    norm_vals = [S[x]/math.sqrt(x) for x in xs]
    mean_norm = float(sum(norm_vals)/len(norm_vals))
    var_norm = float(sum((v-mean_norm)**2 for v in norm_vals)/(len(norm_vals)-1))
    print(f"\nNormalized partial sums at dyadics: mean = {mean_norm:.6e}, var = {var_norm:.6e}")
    print("Last few normalized values (dyadics):")
    for x in xs[-6:]:
        print(f"n={x:12d}  S/sqrt(n) = {S[x]/math.sqrt(x): .6e}")

    # Save some diagnostics to .npz
    np.savez_compressed(out_prefix + "_diag.npz",
                        N=N, nonzero=nonzero, S_N=sN, dy=dy, dy_S=dy_S,
                        beta=beta if beta is not None else np.nan,
                        mean_ratio=mean_ratio, ac=ac, norm_vals=norm_vals)
    print(f"\nDiagnostics saved to {out_prefix}_diag.npz")

    # optional plots
    if do_plots:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib not available:", e)
            return
        # Partial sums plot
        xs_full = np.arange(1, N+1)
        plt.figure(figsize=(10,4))
        plt.plot(xs_full, S[1:], lw=0.5)
        plt.title("S(x) = sum_{n<=x} mu_2(n)")
        plt.xlabel("x"); plt.ylabel("S(x)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_prefix + "_Sx.png", dpi=150)
        # normalized plots
        plt.figure(figsize=(10,4))
        plt.plot(xs_full, S[1:]/xs_full, label="S(x)/x", lw=0.5)
        plt.plot(xs_full, S[1:]/np.sqrt(xs_full), label="S(x)/sqrt(x)", lw=0.5)
        plt.legend(); plt.grid(True)
        plt.title("Normalized partial sums")
        plt.tight_layout()
        plt.savefig(out_prefix + "_norms.png", dpi=150)
        # log-log |S|
        vals_x = np.array(xs)
        vals_y = np.array([abs(S[x]) for x in xs])
        plt.figure(figsize=(6,5))
        plt.loglog(vals_x, vals_y, '.', markersize=4)
        if beta is not None:
            # overlay power law
            Cval = C
            plt.loglog(vals_x, Cval * (vals_x**beta), '-', label=f"fit x^{beta:.3f}")
            plt.legend()
        plt.title("Log-log |S(x)| at dyadic points")
        plt.tight_layout()
        plt.savefig(out_prefix + "_loglog.png", dpi=150)
        print("Plots saved (png).")

# ---------------- Main entry -----------------
def main():
    parser = argparse.ArgumentParser(description="Compute mu2 up to N and run diagnostics")
    parser.add_argument("--N", type=int, default=5000000, help="max n (default 5e6)")
    parser.add_argument("--out", type=str, default="mu2_output.npz", help="output .npz prefix")
    parser.add_argument("--plot", action="store_true", help="produce and save plots")
    args = parser.parse_args()

    N = args.N
    out = args.out
    base = out.replace(".npz","")
    tstart = time.time()
    mu2, S = compute_mu2_up_to(N, show_progress=True)
    # convert to numpy arrays for saving
    mu2_np = np.frombuffer(mu2, dtype=np.int8).copy()  # copy to make contiguous
    # S is already numpy int64
    np.savez_compressed(base + ".npz", mu2=mu2_np, S=S)
    print(f"Saved mu2 and S to {base}.npz")
    run_diagnostics(mu2, S, base, do_plots=args.plot)
    print("Total elapsed:", time.time() - tstart)

if __name__ == "__main__":
    main()
