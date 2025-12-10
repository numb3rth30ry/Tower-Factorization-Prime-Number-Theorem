# tower_parity_corrected.py
# Computes X_p(n) = (1 + omega(m_p)) mod 2, where n = prod p^{m_p}.
# Measures pairwise correlations for p <= Pmax and E[Y], Var[Y] where Y = sum_p X_p.

import math
import time

def primes_upto(n):
    sieve = bytearray(b'\x01') * (n+1)
    sieve[:2] = b'\x00\x00'
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:n+1:step] = b'\x00' * ((n-start)//step + 1)
    return [i for i in range(n+1) if sieve[i]]

def factorize_with_spf(n, spf):
    res = {}
    while n > 1:
        p = spf[n]
        if p == 0 or p == 1:
            res[n] = res.get(n,0) + 1
            break
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        res[p] = cnt
    return res

def sieve_spf(n):
    spf = list(range(n+1))
    if n>=0: spf[0]=0
    if n>=1: spf[1]=1
    for p in range(2, int(n**0.5)+1):
        if spf[p] == p:
            for k in range(p*p, n+1, p):
                if spf[k] == k:
                    spf[k] = p
    return spf

def omega_small(m, spf_small):
    if m <= 1:
        return 0
    tmp = m
    cnt = 0
    while tmp > 1:
        p = spf_small[tmp]
        cnt += 1
        while tmp % p == 0:
            tmp //= p
    return cnt

def is_squarefree(m, spf_small):
    if m <= 1:
        return True
    tmp = m
    while tmp > 1:
        p = spf_small[tmp]
        c = 0
        while tmp % p == 0:
            tmp //= p
            c += 1
            if c >= 2:
                return False
    return True

def compute_stats(N=2_000_000, Pmax=2000):
    t0 = time.time()
    primes_small = primes_upto(Pmax)
    spf = sieve_spf(N)         # for factoring n
    # spf for factoring small exponents: need up to emax ~ log2(N)
    emax = max(64, int(math.log2(max(2,N)))+8)
    spf_small = sieve_spf(emax)

    # initialize counters
    count_p = {p:0 for p in primes_small}     # sum of X_p over n
    count_pq = {(p,q):0 for p in primes_small for q in primes_small if q>p}
    EY = 0
    EY2 = 0

    # main loop
    for n in range(1, N+1):
        f = factorize_with_spf(n, spf)
        # compute X_p for each prime divisor of n (and for p dividing exponent? we only need p that divide n)
        # But also primes that do not divide n contribute 0.
        # X_p = (1 + omega(m_p)) mod 2, where m_p is exponent of p in n
        Xpresent = {}
        for p, m in f.items():
            w = omega_small(m, spf_small)
            xp = (1 + w) & 1
            Xpresent[p] = xp

        # single stats for p <= Pmax
        for p, xp in Xpresent.items():
            if p <= Pmax:
                count_p[p] += xp

        # pairwise stats for small primes in this n
        small_present = [p for p in Xpresent.keys() if p <= Pmax]
        for i in range(len(small_present)):
            for j in range(i+1, len(small_present)):
                p = small_present[i]; q = small_present[j]
                count_pq[(p,q)] += Xpresent[p] * Xpresent[q]

        # total Y (including primes > Pmax)
        Y = sum(Xpresent.values())
        EY += Y
        EY2 += Y*Y

    EY /= N
    EY2 /= N
    VarY = EY2 - EY*EY

    # expectations for small primes
    E_p = {p: count_p[p]/N for p in primes_small}
    E_pq = {(p,q): count_pq[(p,q)]/N for (p,q) in count_pq}
    corr = {(p,q): E_pq[(p,q)] - E_p[p]*E_p[q] for (p,q) in count_pq}

    t1 = time.time()
    print(f"Done N={N}, Pmax={Pmax} in {t1-t0:.2f}s")
    # summary output
    sample_pairs = [(2,3),(2,5),(3,5),(5,7),(11,13),(2,97)]
    print("\nSample pair correlations (X_p = (1+omega(m_p)) mod2):")
    for (p,q) in sample_pairs:
        if (p,q) in corr:
            print(f"Corr({p},{q}) = {corr[(p,q)]:.6e}")
    mean_abs_corr = sum(abs(v) for v in corr.values()) / len(corr)
    print("\nMean absolute correlation (small primes) =", mean_abs_corr)
    print("\nE[Y] =", EY)
    print("Var[Y] =", VarY)
    return dict(EY=EY, VarY=VarY, mean_abs_corr=mean_abs_corr, corr_sample={k:corr[k] for k in sample_pairs if k in corr})

if __name__ == "__main__":
    # you can vary N and Pmax here
    out = compute_stats(N=2_000_000, Pmax=2000)
    print(out)
