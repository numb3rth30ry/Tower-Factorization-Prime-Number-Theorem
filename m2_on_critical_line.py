#!/usr/bin/env python3
"""
m2_on_critical_line.py

Approximate M2(s) = sum_{n>=1} mu2(n) / n^s on the critical line s = 1/2 + i t, t in [0,T].
Methods:
 - truncated Dirichlet sum (sum)
 - smoothed Dirichlet sum with exponential damping (smooth)
 - truncated Euler product (euler)

Usage:
  python m2_on_critical_line.py --mu2npz mu2_5e6.npz --N 5000000 --T 50 --nt 401 --method both --Pmax 20000 --Mmax 40 --plot

Notes:
 - If you don't provide mu2npz, the script will compute mu2 up to N (slower).
 - For stability use smoothed sums; compare with Euler product for consistency.
"""
import argparse, time, math
import numpy as np

# ---------------- utilities ----------------
def primes_upto(n):
    sieve = bytearray(b'\x01')*(n+1)
    sieve[:2]=b'\x00\x00'
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            step=i
            start=i*i
            sieve[start:n+1:step]=b'\x00'*(((n-start)//step)+1)
    return [i for i in range(n+1) if sieve[i]]

def sieve_spf(n):
    spf = list(range(n+1))
    if n>=0: spf[0]=0
    if n>=1: spf[1]=1
    for p in range(2, int(n**0.5)+1):
        if spf[p]==p:
            for k in range(p*p, n+1, p):
                if spf[k]==k:
                    spf[k]=p
    return spf

def is_squarefree_integer(m):
    if m <= 1: return True
    d = 2
    while d*d <= m:
        if m % (d*d) == 0:
            return False
        d += 1
    return True

def omega_small(m):
    if m <= 1: return 0
    cnt = 0
    d = 2
    tmp = m
    while d*d <= tmp:
        if tmp % d == 0:
            cnt += 1
            while tmp % d == 0:
                tmp //= d
        d += 1
    if tmp > 1: cnt += 1
    return cnt

def compute_mu2_up_to(N, show_progress=True):
    spf = sieve_spf(N)
    mu2 = np.zeros(N+1, dtype=np.int8)
    for n in range(1, N+1):
        tmp = n
        ok = True
        total_labels = 0
        while tmp > 1:
            p = spf[tmp]
            e = 0
            while tmp % p == 0:
                tmp //= p
                e += 1
            if not is_squarefree_integer(e):
                ok = False
                break
            w = omega_small(e) if e > 1 else 0
            total_labels += 1 + w
        if not ok:
            mu2[n] = 0
        else:
            mu2[n] = -1 if (total_labels % 2 == 1) else 1
        if show_progress and (n % (max(1, N//8)) == 0 or n==N):
            print(f"[compute_mu2] n={n}/{N}")
    return mu2

# ---------- Dirichlet sum ----------
def dirichlet_sum_grid(mu2, svals, N, smooth=False, Y=None):
    n = np.arange(1, N+1)
    logs = np.log(n)
    if Y is None:
        Y = max(100, N//3)
    res = np.empty(len(svals), dtype=np.complex128)
    for i,s in enumerate(svals):
        t = s.imag
        if smooth:
            factor = np.exp(-n / Y)
        else:
            factor = 1.0
        exponents = -s.real * logs - 1j * t * logs
        terms = mu2[1:N+1].astype(np.complex128) * np.exp(exponents) * factor
        res[i] = terms.sum()
    return res

# ---------- Euler product ----------
def mu_m_values_up_to(Mmax):
    mu = [1]*(Mmax+1)
    if Mmax >= 0: mu[0] = 0
    if Mmax >= 1: mu[1] = 1
    lp = [0]*(Mmax+1)
    primes = []
    for i in range(2, Mmax+1):
        if lp[i]==0:
            lp[i]=i
            primes.append(i)
        for p in primes:
            if p > lp[i] or i*p > Mmax: break
            lp[i*p] = p
    mu = [0]*(Mmax+1)
    mu[1]=1
    for m in range(2, Mmax+1):
        tmp = m
        squarefree = True
        distinct = 0
        while tmp > 1:
            p = lp[tmp] if lp[tmp] != 0 else tmp
            cnt = 0
            while tmp % p == 0:
                tmp //= p
                cnt += 1
            if cnt >= 2:
                squarefree = False
                break
            distinct += 1
        if not squarefree:
            mu[m] = 0
        else:
            mu[m] = -1 if (distinct % 2 == 1) else 1
    return mu

def euler_product_grid(svals, Pmax, Mmax, mu_m_cache=None):
    if mu_m_cache is None or len(mu_m_cache) <= Mmax:
        mu_m_cache = mu_m_values_up_to(Mmax)
    primes = primes_upto(Pmax)
    out = np.empty(len(svals), dtype=np.complex128)
    for i,s in enumerate(svals):
        prod = 1.0 + 0j
        for p in primes:
            term = 1.0 + 0j
            p_pow = p**(-s)
            for m in range(1, Mmax+1):
                mu_m = mu_m_cache[m]
                if mu_m == 0: continue
                term += mu_m * (p_pow ** m)
            prod *= term
        out[i] = prod
    return out

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Approximate M2(1/2 + i t) on t in [0,T]")
    parser.add_argument("--mu2npz", type=str, default="", help="npz file with 'mu2' array")
    parser.add_argument("--N", type=int, default=2000000, help="truncate Dirichlet sum at N")
    parser.add_argument("--T", type=float, default=50.0, help="max t")
    parser.add_argument("--nt", type=int, default=401, help="number of t points")
    parser.add_argument("--method", choices=["sum","smooth","euler","both"], default="both")
    parser.add_argument("--Pmax", type=int, default=20000, help="max prime for Euler product")
    parser.add_argument("--Mmax", type=int, default=40, help="max exponent for Euler product")
    parser.add_argument("--plot", action="store_true", help="save plots")
    args = parser.parse_args()

    # load or compute mu2
    if args.mu2npz:
        try:
            d = np.load(args.mu2npz)
            mu2 = d['mu2']
            N_avail = len(mu2)-1
            if args.N > N_avail:
                print(f"[warning] requested N={args.N} but mu2 npz has {N_avail}, using {N_avail}")
                N = N_avail
            else:
                N = args.N
            print(f"Loaded mu2 up to {len(mu2)-1} from {args.mu2npz}")
        except Exception as e:
            print("Failed to load mu2 npz:", e)
            print("Computing mu2 up to N (this may be slow)...")
            N = args.N
            mu2 = compute_mu2_up_to(N)
    else:
        N = args.N
        mu2 = compute_mu2_up_to(N)

    # set up s-grid on critical line
    tvals = np.linspace(0.0, args.T, args.nt)
    svals = [0.5 + 1j*t for t in tvals]

    results = {}
    tstart = time.time()
    if args.method in ("sum","both"):
        print("[compute] truncated Dirichlet sum (no smoothing)...")
        res_sum = dirichlet_sum_grid(mu2, svals, N, smooth=False)
        results['sum'] = res_sum
    if args.method in ("smooth","both"):
        print("[compute] smoothed Dirichlet sum (Y=N/3)...")
        Y = max(100, N//3)
        res_smooth = dirichlet_sum_grid(mu2, svals, N, smooth=True, Y=Y)
        results['smooth'] = res_smooth
    if args.method in ("euler","both"):
        print(f"[compute] Euler product Pmax={args.Pmax}, Mmax={args.Mmax} ...")
        mu_m_cache = mu_m_values_up_to(args.Mmax)
        res_euler = euler_product_grid(svals, args.Pmax, args.Mmax, mu_m_cache)
        results['euler'] = res_euler
    tstop = time.time()
    print(f"[compute] done in {tstop-tstart:.2f}s")

    # show table: t | |sum| | |smooth| | |euler|
    print("\n t  |  |sum|  |  |smooth|  |  |euler|")
    for i,t in enumerate(tvals):
        line = f"{t:6.3f} |"
        line += f" { (abs(results['sum'][i]) if 'sum' in results else float('nan')):10.6e} |"
        line += f" { (abs(results['smooth'][i]) if 'smooth' in results else float('nan')):10.6e} |"
        line += f" { (abs(results['euler'][i]) if 'euler' in results else float('nan')):10.6e}"
        print(line)

    # mean-square and running mean-square
    for key,arr in results.items():
        magsq = np.abs(arr)**2
        mean_sq = magsq.mean()
        running_mean = np.cumsum(magsq) / np.arange(1, len(magsq)+1)
        print(f"[result] method={key} mean-square (grid avg) = {mean_sq:.6e}")
        print(f"[result] method={key} running I(T) final = {running_mean[-1]:.6e}")

    # optional plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib required for plotting:", e)
            return
        plt.figure(figsize=(10,4))
        for key,arr in results.items():
            plt.plot(tvals, np.abs(arr), label=f'|M2| ({key})')
        plt.yscale('log')
        plt.xlabel('t (s = 1/2 + i t)')
        plt.ylabel('|M2(1/2+it)| (log scale)')
        plt.legend(); plt.grid(True, which='both', ls='--', alpha=0.6)
        plt.savefig('M2_critical_abs.png', dpi=200)
        print("Saved M2_critical_abs.png")

        plt.figure(figsize=(10,4))
        for key,arr in results.items():
            magsq = np.abs(arr)**2
            running_mean = np.cumsum(magsq) / np.arange(1, len(magsq)+1)
            plt.plot(tvals, running_mean, label=f'I(t) ({key})')
        plt.xlabel('t'); plt.ylabel('running mean-square I(t)'); plt.legend(); plt.grid(True)
        plt.savefig('M2_critical_running_mean.png', dpi=200)
        print("Saved M2_critical_running_mean.png")

    print("Total elapsed:", time.time() - tstart)

if __name__ == "__main__":
    main()
