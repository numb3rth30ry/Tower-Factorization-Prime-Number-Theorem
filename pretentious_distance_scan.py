#!/usr/bin/env python3
"""
pretentious_distance_scan.py

Compute pretentious distance D_P(t) = sum_{p <= P} (1 - Re(mu2(p) p^{-it}))/p
for a grid of t in [0, T] and for a list of P cutoffs. Summarize growth of min/mean/max D_P(t).

Usage examples:
  python pretentious_distance_scan.py --mu2npz mu2_5e6.npz --Pmax_list 100,1000,5000,20000 --T 100 --nt 400 --plot
  python pretentious_distance_scan.py --Pmax_list 100,1000,5000,20000 --T 50 --nt 201

Requirements: numpy, matplotlib (optional for plotting). Uses only standard library + numpy.

Outputs:
 - printed table: for each P: min D, median D, mean D, max D
 - optional plots: D_vs_P.png (min/mean/max vs P), heatmap_D_t_P.png
"""
import argparse, math, time
import numpy as np

# ---------------- helpers ----------------
def primes_upto(n):
    sieve = bytearray(b'\x01')*(n+1)
    sieve[:2] = b'\x00\x00'
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:n+1:step] = b'\x00'*(((n-start)//step)+1)
    return [i for i in range(n+1) if sieve[i]]

def load_mu2_if_possible(npzfile, needed_max_p):
    if not npzfile:
        return None
    try:
        d = np.load(npzfile)
        mu2 = d['mu2']
        max_provided = len(mu2)-1
        if max_provided < needed_max_p:
            print(f"[warning] mu2 npz contains mu2 up to {max_provided}, but we need primes up to {needed_max_p}.")
            print("Proceeding but mu2[p] for p > provided will be set to -1 (prime leaf).")
        return mu2
    except Exception as e:
        print("Failed to load mu2 npz:", e)
        return None

# ---------------- main computation ----------------
def compute_D_grid(mu2_npz, P_list, T, nt, t_min=0.0):
    # prepare t grid
    tvals = np.linspace(t_min, T, nt)
    # primes up to max P
    Pmax_all = max(P_list)
    primes = primes_upto(Pmax_all)
    prime_set = set(primes)

    # load mu2 if available, otherwise assume mu2[p] = -1 for primes (primes are leaves)
    mu2 = load_mu2_if_possible(mu2_npz, Pmax_all)

    # prepare f(p) = mu2(p) as complex (if mu2 not available, take -1)
    f_at_prime = {}
    for p in primes:
        if mu2 is not None and p < len(mu2):
            f_at_prime[p] = complex(int(mu2[p]))
        else:
            f_at_prime[p] = -1.0 + 0j

    # compute D for each P in P_list: we will build cumulative sums across primes to be efficient
    # order primes ascending; for each prime we compute contribution to D(t) = (1 - Re(f_p * p^{-it})) / p
    # accumulate for progressively larger P.
    t_complex = 1.0j * tvals  # array
    # Precompute p^{-it} = exp(-i t log p) factors per prime and t
    print("[info] computing p^{-it} factors for each prime (this may take a second)...")
    logp = {p: math.log(p) for p in primes}
    # We'll compute contributions prime-by-prime and store D for each desired P checkpoint.
    D_t_current = np.zeros(nt, dtype=np.float64)
    D_results = {}  # P -> D_t array
    primes_sorted = primes  # ascending

    P_set = set(P_list)
    next_checkpoints = sorted(P_list)

    p_index = 0
    for p in primes_sorted:
        # compute p^{-it} for all t: exp(-i t log p)
        phase = np.exp(-1j * (tvals * logp[p]))  # vector of length nt
        fp = f_at_prime[p]
        # contribution = (1 - Re(fp * phase)) / p
        contrib = (1.0 - (fp * phase).real) / p
        D_t_current += contrib
        p_index += 1
        # if p is a checkpoint, store snapshot
        while next_checkpoints and p >= next_checkpoints[0]:
            P_now = next_checkpoints.pop(0)
            D_results[P_now] = D_t_current.copy()
    # in case some P larger than largest prime? handled by primes upto Pmax_all so ok
    return tvals, D_results

def summarize_D_results(tvals, D_results):
    # For each P, compute min, median, mean, max over t
    summary = []
    for P in sorted(D_results.keys()):
        arr = D_results[P]
        mn = float(arr.min())
        med = float(np.median(arr))
        mean = float(arr.mean())
        mx = float(arr.max())
        summary.append((P, mn, med, mean, mx))
    return summary

def print_summary_table(summary):
    print("\nP\tmin(D)\tmedian(D)\tmean(D)\tmax(D)")
    for row in summary:
        P, mn, med, mean, mx = row
        print(f"{P}\t{mn:.6e}\t{med:.6e}\t{mean:.6e}\t{mx:.6e}")

# ---------------- plotting ----------------
def plot_results(tvals, D_results, summary, out_prefix="pretentious"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available:", e)
        return
    Ps = sorted(D_results.keys())
    # plot min/mean/max vs P
    mins = [s[1] for s in summary]
    means = [s[3] for s in summary]
    maxs = [s[4] for s in summary]
    plt.figure(figsize=(8,4))
    plt.plot(Ps, mins, '-o', label='min D(t)')
    plt.plot(Ps, means, '-o', label='mean D(t)')
    plt.plot(Ps, maxs, '-o', label='max D(t)')
    plt.xscale('log')
    plt.xlabel('P (prime cutoff)')
    plt.ylabel('D_P(t)')
    plt.title('Pretentious distance statistics vs P')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_prefix + "_D_vs_P.png", dpi=200)
    print(f"Saved {out_prefix}_D_vs_P.png")

    # heatmap t x P (for reasonably small set of P)
    # stack arrays in increasing P order
    arr_stack = np.vstack([D_results[P] for P in Ps])
    # arr_stack shape len(Ps) x nt ; we want nt x len(Ps) for imshow with t vertical
    plt.figure(figsize=(10,6))
    plt.imshow(arr_stack.T, aspect='auto', origin='lower',
               extent=[min(Ps), max(Ps), tvals[0], tvals[-1]], cmap='viridis')
    plt.colorbar(label='D_P(t)')
    plt.xscale('log')
    plt.xlabel('P (prime cutoff)')
    plt.ylabel('t')
    plt.title('Heatmap of D_P(t) (t vertical, P horizontal)')
    plt.tight_layout()
    plt.savefig(out_prefix + "_heatmap_D_t_P.png", dpi=200)
    print(f"Saved {out_prefix}_heatmap_D_t_P.png")

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Scan pretentious distance D_P(t) for mu2")
    parser.add_argument("--mu2npz", type=str, default="", help="Path to mu2 npz (optional)")
    parser.add_argument("--Pmax_list", type=str, default="100,1000,5000,20000", help="comma list of P cutoffs")
    parser.add_argument("--T", type=float, default=100.0, help="max t")
    parser.add_argument("--nt", type=int, default=401, help="number of t grid points")
    parser.add_argument("--plot", action="store_true", help="save plots")
    args = parser.parse_args()

    P_list = [int(x) for x in args.Pmax_list.split(',') if x.strip()]
    P_list = sorted(set(P_list))
    print(f"[info] P_list = {P_list}, T = {args.T}, nt = {args.nt}")

    t0 = time.time()
    tvals, D_results = compute_D_grid(args.mu2npz, P_list, args.T, args.nt)
    summary = summarize_D_results(tvals, D_results)
    print_summary_table(summary)
    if args.plot:
        plot_results(tvals, D_results, summary)
    print(f"[done] elapsed {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
