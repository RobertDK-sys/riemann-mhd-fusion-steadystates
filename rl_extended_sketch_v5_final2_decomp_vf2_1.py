"""
rl_extended_sketch_v5_final2_decomp_vf2_1.py  (autonome -- aucun import local)
========================================================================================

Version autonome (voir Algorithme_A_decomp_vf2_1.py pour le detail des
etapes desormais inlinees : equation caracteristique + resolution de
s2(t) par brentq -- "Algorithme 2" -- calcul de b(t,k), chainage vf2_1,
decomposition complete de chaque (N1,b1)/(N2,b2), construction du
vecteur composite b[1..M]).

build_fb_lattice_from_chain() remplace build_fb_lattice(s2_t, n_b, ...) :
fb = un ECHANTILLON Monte-Carlo de paires du vecteur composite b[1..M]
issu de la chaine (voir sample_fb_pairs, note echelle ci-dessous), au
lieu des combinaisons croissantes de k dans range(-n_b,n_b). Tout le
reste du fichier -- Ef dynamique (EWMA), ReferenceTable,
AssociationIndex, regle d'eligibilite de la triangularite negative,
NBISequencingEnvV5, Q-learning, entrainement/evaluation
multi-dispositifs, diagnostic sort_mismatch -- est CONSERVE A
L'IDENTIQUE : ces elements ne consomment qu'une liste plate fb_lattice,
quelle que soit la maniere dont elle a ete construite.

NOTE ECHELLE / ECHANTILLONNAGE : CHAIN_N_MAX = N_MAX = 65536 = 2**16
(echelle REELLE de N1/N2, comme Algorithme_A/B_decomp_vf2_1.py). fb
est un echantillon Monte-Carlo de PAIR_SAMPLE_SIZE paires (au lieu de
l'enumeration exhaustive O(M^2), infaisable a cette echelle) --
PAIR_SAMPLE_SIZE est ici volontairement plus modeste que dans
Algorithme_A/B_decomp_vf2_1.py car fb_lattice est interroge des
dizaines de milliers de fois pendant l'entrainement RL.

NOTE VARIANTES : rl_extended_sketch_v5_final_decomp_vf2_1.py ne differait de ce fichier QUE par
le calcul interne de b[] (boucle de signe supplementaire). Cette
distinction disparait avec la decomposition (les candidats de la chaine
portent deja un signe) -- les deux fichiers "_decomp_vf2_1" utilisent
donc la meme fonction build_fb_lattice_from_chain et produisent le meme
comportement.
"""
import math
import random
import numpy as np
import random as _random
import numpy as np
from math import asin, log, pi
from fractions import Fraction as Frac
from bisect import bisect_left
from scipy.optimize import brentq

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict

random.seed(0)


# =============================================================================
# 0. EQUATION CARACTERISTIQUE ET RESOLUTION DE s2(t)  (Algorithme 2 fourni)
#    0 = |8*s2^6 - 24*s2^4 + 24*s2^2 - 8|^(2/3)
#        * (24*k3^2*s2^6 - 72*k3^2*s2^4 - 24*k3^2) + 4*s2^4,   avec k3 = -t
#    C'est ICI, et seulement ici, que s2(t) est resolue numeriquement
#    (brentq) -- il n'y a plus de table figee de valeurs de s2.
# =============================================================================
def _equation(s2, k3):
    A = 8 * s2 ** 6 - 24 * s2 ** 4 + 24 * s2 ** 2 - 8
    return (abs(A) ** (2 / 3)) * (24 * k3 ** 2 * s2 ** 6 - 72 * k3 ** 2 * s2 ** 4 - 24 * k3 ** 2) + 4 * s2 ** 4


def get_s2(t):
    """Racine reelle s2(t) de l'equation caracteristique (nouvelle equation,
    "Algorithme 2") pour k3=-t, cherchee par dichotomie (brentq). Bracket et
    xtol repris des fichiers de chainage d'origine (algorithme_chainage_beta_
    vf2_1/vf2_2_avec_decomposition (vf1-2).py), plus larges que le
    (-0.9999,-0.0001) de l'Algorithme 2 fourni en exemple : ce dernier
    suffit jusque t~500-800 mais brentq echoue au-dela (plus de changement
    de signe dans ce bracket etroit une fois s2 tres proche de -1), or on
    veut ici retrouver t_max=2000 comme dans les fichiers de chainage
    d'origine."""
    return brentq(_equation, -0.999999999999, -1e-12, args=(-t,), xtol=1e-15)


LN2 = log(2)
N_MAX = 65536


def generate_candidates(t_max, k_range, verbose=True):
    """Genere le pool de candidats {(t,k,signe,b,s2)} avec
        b(t,k) = (asin(signe*|s2(t)|) - 2*k*pi) / ln(2),
    k entier non nul dans [-k_range,k_range], signe dans {+1,-1},
    t = 1..t_max. C'EST ICI que TOUS les b[l] de l'algorithme sont
    calcules -- il n'y a plus de calcul de b ailleurs dans le fichier.
    s2(t) est calcule UNE SEULE FOIS par t et stocke dans le candidat
    (5e champ) pour eviter de le re-resoudre (brentq) plus tard, en
    particulier lors de la construction des listes de decomposition ou
    Nf peut atteindre N_MAX=65536 candidats a relire."""
    if verbose:
        print(f"  [generate_candidates] demarrage : t_max={t_max}, k_range={k_range}", flush=True)
    cands = []
    for t in range(1, t_max + 1):
        s2 = get_s2(t)
        for sign in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k != 0:
                    b = (asin(sign * abs(s2)) - 2 * k * pi) / LN2
                    cands.append((t, k, sign, b, s2))
    if verbose:
        print(f"  [generate_candidates] termine : {len(cands)} candidats", flush=True)
    return cands


X0_LIST = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
           13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
           32.787, 33.596]


def find_between(lo, hi, pool):
    lo, hi = min(lo, hi), max(lo, hi)
    return [c for c in pool if lo < c[3] < hi]


class FastPool:
    """Pre-indexe le pool (tries par signe) pour des recherches rapides
    par dichotomie au lieu de filtrer/trier l'integralite du pool a
    chaque appel."""
    def __init__(self, candidates):
        self.pos = sorted([(b, t, k) for (t, k, sign, b, s2) in candidates if sign == 1])
        self.neg = sorted([(b, t, k) for (t, k, sign, b, s2) in candidates if sign == -1])
        self.pos_b = [c[0] for c in self.pos]
        self.neg_b = [c[0] for c in self.neg]

    def nearest(self, sign_needed, x0, side_greater, k=10):
        lst, bs = (self.pos, self.pos_b) if sign_needed == 1 else (self.neg, self.neg_b)
        i = bisect_left(bs, x0)
        if side_greater:
            window = lst[i:i + k]
        else:
            window = lst[max(0, i - k):i][::-1]
        return [(t, kk, b) for (b, t, kk) in window]

# =============================================================================
# 1. CHAINAGE -- GROUPE 1 : "b HERITE, POIDS CONTRAINT"
#    Etape 1 : recherche LIBRE de (N1,N2). Etape n>=2 : b1_n = b2_(n-1)
#    (position heritee), N1_n libre mais N1_n != N2_(n-1) et
#    N1_n + N2_(n-1) <= N_MAX.
# =============================================================================
def best_rational(ratio, n_max):
    if ratio <= 0:
        return None
    invert = ratio > 1
    x = 1 / ratio if invert else ratio
    fx = Frac(x).limit_denominator(n_max)
    p, q = fx.numerator, fx.denominator
    if p == 0:
        p = 1
    if invert:
        p, q = q, p
    if p > n_max or q > n_max:
        return None
    return p, q


def best_rational_bounded(ratio, n1_max, n2_max):
    """Bornes independantes 1<=p<=n1_max, 1<=q<=n2_max via l'arbre de
    Stern-Brocot (methode des mediantes)."""
    if ratio <= 0 or n1_max < 1 or n2_max < 1:
        return None
    a_num, a_den = 0, 1
    b_num, b_den = 1, 0
    for _ in range(200):
        m_num, m_den = a_num + b_num, a_den + b_den
        if m_num > n1_max or m_den > n2_max:
            break
        cross = m_num - ratio * m_den
        if abs(cross) < 1e-15:
            a_num, a_den = m_num, m_den
            b_num, b_den = m_num, m_den
            break
        elif cross < 0:
            a_num, a_den = m_num, m_den
        else:
            b_num, b_den = m_num, m_den
    candidates = []
    if a_den > 0 and a_num <= n1_max and a_den <= n2_max:
        candidates.append((a_num, a_den))
    if b_den > 0 and b_num <= n1_max and b_den <= n2_max:
        candidates.append((b_num, b_den))
    if not candidates:
        return None
    return min(candidates, key=lambda pq: abs(pq[0] / pq[1] - ratio))


def best_partner_free(anchor_b, anchor_sign, target_x0, fp, n_max=N_MAX, k_near=10):
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)
    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        fr = best_rational(ratio, n_max)
        if fr is None:
            continue
        N1, N2 = fr
        xc = (N1 * anchor_b + N2 * b2) / (N1 + N2)
        err = abs(xc - target_x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign_needed, b2, N1, N2, xc)
    return best


def best_partner_constrained(anchor_b, anchor_sign, target_x0, prev_N2, fp, n_max=N_MAX, k_near=10):
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)
    n1_max = n_max - prev_N2
    if n1_max < 1:
        return None
    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        fr = best_rational_bounded(ratio, n1_max, n_max)
        if fr is None:
            continue
        N1, N2 = fr
        if N1 == prev_N2:
            alt = []
            for N1c in (N1 - 1, N1 + 1):
                if 1 <= N1c <= n1_max and N1c != prev_N2 and ratio > 0:
                    N2c = round(N1c / ratio)
                    N2c = max(1, min(n_max, N2c))
                    alt.append((N1c, N2c))
            if not alt:
                continue
            N1, N2 = min(alt, key=lambda pq: abs((pq[0] * anchor_b + pq[1] * b2) / (pq[0] + pq[1]) - target_x0))
        xc = (N1 * anchor_b + N2 * b2) / (N1 + N2)
        err = abs(xc - target_x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign_needed, b2, N1, N2, xc)
    return best


def run_chain(start_idx, x0_list, pool, fp, n_max=N_MAX, max_chains=None):
    """max_chains : ne calcule la chaine que pour les max_chains premiers
    ancrages (au lieu de TOUS les ancrages de l'intervalle) -- utile a
    l'echelle N_MAX=65536 ou construire une chaine par ancrage est couteux
    et on n'a besoin, en pratique, que d'une seule chaine (CHAIN_RANK)."""
    x0_1, x0_2 = x0_list[start_idx], x0_list[start_idx + 1]
    remaining_targets = x0_list[start_idx + 1:]
    initial_anchors = find_between(x0_1, x0_2, pool)
    anchors_to_process = initial_anchors if max_chains is None else initial_anchors[:max_chains]
    all_chains = []
    for l, (t0, k0, sign0, b0, s2_0) in enumerate(anchors_to_process, start=1):
        chain = [{"step": 0, "t": t0, "k": k0, "sign": sign0, "b": b0,
                  "target": None, "N1": None, "N2": None, "err": None}]
        current_b, current_sign = b0, sign0
        prev_N2 = None
        for step, target in enumerate(remaining_targets, start=1):
            if step == 1:
                res = best_partner_free(current_b, current_sign, target, fp, n_max)
            else:
                res = best_partner_constrained(current_b, current_sign, target, prev_N2, fp, n_max)
            if res is None:
                break
            err, t2, k2, sign2, b2, N1, N2, xc = res
            chain.append({"step": step, "t": t2, "k": k2, "sign": sign2, "b": b2,
                          "target": target, "N1": N1, "N2": N2, "err": err})
            current_b, current_sign = b2, sign2
            prev_N2 = N2
        all_chains.append({"l": l, "start": (t0, k0, sign0, b0), "chain": chain,
                            "n_l": chain[-1]["step"]})
    return x0_1, x0_2, remaining_targets, initial_anchors, all_chains

# =============================================================================
# 2. DECOMPOSITION Nf*b_f = sum(b_k) -- pour chaque (N1,b1)/(N2,b2) d'une
#    chaine, retrouve Nf triplets DISTINCTS {t_k,sign_k,k_k,s2_val_k,b_k}
#    dont la somme approche Nf*b_f (recuit simule, cout incremental).
# =============================================================================
def generate_decomposition_pool(t_max, k_range):
    """Pool DEDIE a la decomposition (independant du pool de recherche de
    chaine) -- utilise generate_candidates ci-dessus (meme equation)."""
    return generate_candidates(t_max=t_max, k_range=k_range, verbose=False)


def decompose_bf_into_bk(Nf, b_f_target, decomp_pool, iters=20000, seed=0, exclude=None):
    _random.seed(seed)
    n = len(decomp_pool)
    exclude = exclude or set()
    pool_idx = [i for i in range(n) if i not in exclude]
    assert len(pool_idx) >= Nf, (
        f"pas assez de candidats ({len(pool_idx)}) pour Nf={Nf} -- "
        f"augmentez t_max/k_range du pool de decomposition"
    )
    bs = [c[3] for c in decomp_pool]
    idx = _random.sample(pool_idx, Nf)
    idx_set = set(idx)
    current_sum = sum(bs[i] for i in idx)
    target_sum = Nf * b_f_target
    cur_cost = abs(target_sum - current_sum)
    best_cost, best_idx = cur_cost, list(idx)

    T0, T1 = 5.0, 1e-4
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        pos = _random.randrange(Nf)
        old_i = idx[pos]
        trial = _random.choice(pool_idx)
        tries = 0
        while trial in idx_set and tries < 20:
            trial = _random.choice(pool_idx)
            tries += 1
        if trial in idx_set:
            continue
        new_sum = current_sum - bs[old_i] + bs[trial]
        new_cost = abs(target_sum - new_sum)
        if new_cost < cur_cost or _random.random() < 2.718281828 ** (-(new_cost - cur_cost) / max(T, 1e-12)):
            idx[pos] = trial
            idx_set.discard(old_i); idx_set.add(trial)
            current_sum = new_sum
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost, best_idx = cur_cost, list(idx)

    bk_list = [{"b_k": decomp_pool[i][3], "sign_k": decomp_pool[i][2],
                "s2_val_k": decomp_pool[i][4], "k_k": decomp_pool[i][1],
                "t_k": decomp_pool[i][0]} for i in best_idx]
    return best_cost, bk_list


def annotate_chain_with_decompositions(chain, decomp_pool, iters=20000, seed=0):
    for step_idx in range(1, len(chain)):
        row = chain[step_idx]
        anchor = chain[step_idx - 1]
        idx_a = next((j for j, c in enumerate(decomp_pool)
                      if c[:3] == (anchor["t"], anchor["k"], anchor["sign"])), None)
        idx_r = next((j for j, c in enumerate(decomp_pool)
                      if c[:3] == (row["t"], row["k"], row["sign"])), None)
        excl_a = {idx_a} if idx_a is not None else set()
        excl_r = {idx_r} if idx_r is not None else set()
        res1, bk1 = decompose_bf_into_bk(row["N1"], anchor["b"], decomp_pool, iters, seed, excl_a)
        res2, bk2 = decompose_bf_into_bk(row["N2"], row["b"], decomp_pool, iters, seed + 1, excl_r)
        row["decomp_b1"] = {"Nf": row["N1"], "b_f": anchor["b"], "residu": res1, "b_k_list": bk1}
        row["decomp_b2"] = {"Nf": row["N2"], "b_f": row["b"], "residu": res2, "b_k_list": bk2}
    return chain


# =============================================================================
# 3. VECTEUR COMPOSITE b[1..M] -- concatenation, dans l'ordre de la chaine,
#    de la decomposition de (N1,b1) PUIS (N2,b2), etape apres etape.
# =============================================================================
def build_composite_b(chain, max_b_total=None):
    """Concatene, dans l'ordre de la chaine, les b_k de decomp_b1 PUIS
    decomp_b2 de chaque etape. max_b_total est desormais un simple
    garde-fou MEMOIRE (le vecteur b[] est garde tel quel en RAM, plus
    d'enumeration exhaustive des paires en aval -- voir sample_fb_pairs
    ci-dessous) ; None = pas de limite."""
    for row in chain:
        if row["step"] >= 1 and ("decomp_b1" not in row or "decomp_b2" not in row):
            raise ValueError("chaine non annotee -- appelez d'abord annotate_chain_with_decompositions().")
    b_composite = []
    for row in chain:
        if row["step"] == 0:
            continue
        for key in ("decomp_b1", "decomp_b2"):
            for bk in row[key]["b_k_list"]:
                b_composite.append(bk["b_k"])
    if max_b_total is not None and len(b_composite) > max_b_total:
        raise ValueError(
            f"Vecteur composite b[] de taille {len(b_composite)} > max_b_total={max_b_total}."
        )
    return b_composite


def sample_fb_pairs(b, n_samples, seed=0):
    """Echantillonne n_samples paires DISTINCTES (i1 != i2) parmi les
    M(M-1)/2 paires possibles du vecteur composite b[], et renvoie les
    differences b[i1]-b[i2] -- remplace l'enumeration EXHAUSTIVE des
    paires (devenue infeasable en O(M^2) des que M approche N_MAX=65536)
    par un echantillon Monte-Carlo represantatif de meme nature (une
    difference par paire tiree au hasard dans le vecteur composite)."""
    rng = np.random.default_rng(seed)
    b_arr = np.asarray(b, dtype=np.float64)
    M = len(b_arr)
    i1 = rng.integers(0, M, size=n_samples)
    i2 = rng.integers(0, M, size=n_samples)
    mask = i1 == i2
    while mask.any():
        n_bad = int(mask.sum())
        i1[mask] = rng.integers(0, M, size=n_bad)
        i2[mask] = rng.integers(0, M, size=n_bad)
        mask = i1 == i2
    return (b_arr[i1] - b_arr[i2])

# ---------------------------------------------------------------------------
# Parametres de chaine / decomposition (voir Algorithme_A_decomp_vf2_1.py)
# ---------------------------------------------------------------------------
CHAIN_T_MAX, CHAIN_K_RANGE = 2000, 20   # repris des fichiers de chainage d'origine
CHAIN_N_MAX = N_MAX                     # = 65536 = 2**16 -- echelle reelle de N1/N2
INTERVAL_INDEX = 1
CHAIN_RANK = 0

DECOMP_T_MAX, DECOMP_K_RANGE = 2000, 20 # repris des fichiers de chainage d'origine
DECOMP_ITERS = 300000                   # iterations du recuit simule (valeur des fichiers de reference)

PAIR_SAMPLE_SIZE = 50_000               # plus modeste qu'Algorithme_A/B_decomp : fb_lattice
                                         # est interroge des dizaines de milliers de fois
                                         # pendant l'entrainement RL -- le garder modeste
                                         # evite un entrainement trop lent.


def build_fb_lattice_from_chain(interval_index=INTERVAL_INDEX, chain_rank=CHAIN_RANK,
                                 chain_n_max=CHAIN_N_MAX, chain_t_max=CHAIN_T_MAX,
                                 chain_k_range=CHAIN_K_RANGE, decomp_t_max=DECOMP_T_MAX,
                                 decomp_k_range=DECOMP_K_RANGE, decomp_iters=DECOMP_ITERS,
                                 pair_sample_size=PAIR_SAMPLE_SIZE, verbose=True):
    """Remplace build_fb_lattice(s2_t, n_b, ...) : construit le fb-lattice a
    partir du vecteur composite b[1..M] issu des decompositions completes
    successives (N1,b1) -> (N2,b2) -> ... d'UNE chaine trouvee sur
    l'intervalle [X0_LIST[interval_index], X0_LIST[interval_index+1]]."""
    if verbose:
        print("  [build_fb_lattice_from_chain] 1. pool de recherche de chaîne...", flush=True)
    candidates = generate_candidates(t_max=chain_t_max, k_range=chain_k_range, verbose=False)
    fp = FastPool(candidates)

    if verbose:
        print(f"  [build_fb_lattice_from_chain] 2. recherche de chaîne sur "
              f"l'intervalle [{interval_index}] (n_max={chain_n_max})...", flush=True)
    x0_1, x0_2, remaining, anchors, chains = run_chain(
        interval_index, X0_LIST, candidates, fp, n_max=chain_n_max, max_chains=chain_rank + 1
    )
    if not chains:
        raise ValueError(f"Aucune chaîne trouvée sur l'intervalle [{interval_index}] ({x0_1}, {x0_2}).")
    chain = chains[chain_rank]["chain"]
    if verbose:
        print(f"  [build_fb_lattice_from_chain] -> chaîne #{chain_rank} : "
              f"{chains[chain_rank]['n_l']} étape(s) sur {len(remaining)} cible(s).", flush=True)

    if verbose:
        print("  [build_fb_lattice_from_chain] 3. pool de décomposition...", flush=True)
    decomp_pool = generate_decomposition_pool(t_max=decomp_t_max, k_range=decomp_k_range)

    if verbose:
        print("  [build_fb_lattice_from_chain] 4. décomposition COMPLETE de la chaîne...", flush=True)
    annotate_chain_with_decompositions(chain, decomp_pool, iters=decomp_iters, seed=0)

    b = build_composite_b(chain)
    if verbose:
        M = len(b)
        print(f"  [build_fb_lattice_from_chain] -> vecteur composite b[] : {M:,} valeurs "
              f"({M*(M-1)//2:,} paires possibles)", flush=True)
        print(f"  [build_fb_lattice_from_chain] 5. échantillonnage de "
              f"{pair_sample_size:,} paires...", flush=True)

    fb = sample_fb_pairs(b, pair_sample_size, seed=1)
    return fb.tolist()


@dataclass
class DeviceParams:
    name: str
    B: float
    Ip: float
    R: float
    a: float
    dt: float = 10.0

    @property
    def l(self) -> float:
        return 2 * math.pi * self.R

    @property
    def A(self) -> float:
        return math.pi * self.a ** 2


EAST = DeviceParams(name="EAST", B=2.75, Ip=330_000, R=1.91, a=0.45)
W7X = DeviceParams(name="Wendelstein 7-X", B=3.0, Ip=86_500, R=5.5, a=0.53)
DIIID = DeviceParams(name="DIII-D", B=2.2, Ip=1_500_000, R=1.67, a=0.67)

mu0 = 12.566370614e-7
Ef_REF = 5.0

s2_sample = [
    -0.996731716, -0.99653979777, -0.99632393409, -0.99607934533,
    -0.99579988915, -0.99547753758, -0.99510159327, -0.99465747946,
    -0.99412479292, -0.99347415065, -0.9926614346, -0.99161749897,
    -0.9902273088, -0.98828435414, -0.98537717072, -0.98055121855,
    -0.97097209774, -0.94282328841,
]



def magnetic_energy_power(device: DeviceParams) -> float:
    E_stored_joules = (device.A * device.B ** 2 * device.l) / (2 * mu0)
    return E_stored_joules / device.dt / 1e6


def nbi_wave_power(i: int, pulse_period: int = 50, ramp: int = 10,
                    nbi_peak: float = 3.0, p_rf_baseline: float = 1.65) -> float:
    phase = i % pulse_period
    hold = max(pulse_period - 2 * ramp, 0)
    if phase < ramp:
        nbi = nbi_peak * (phase / ramp)
    elif phase < ramp + hold:
        nbi = nbi_peak
    elif phase < 2 * ramp + hold:
        nbi = nbi_peak * (1 - (phase - ramp - hold) / ramp)
    else:
        nbi = 0.0
    return p_rf_baseline + nbi


def device_heating_scale(device: DeviceParams, reference_device: DeviceParams = EAST) -> float:
    return magnetic_energy_power(device) / magnetic_energy_power(reference_device)


def make_device_source(device: DeviceParams, pulse_period: int = 50, ramp: int = 10,
                        nbi_peak: float = 3.0, p_rf_baseline: float = 1.65,
                        auto_scale_heating: bool = True
                        ) -> Callable[[int], Tuple[float, float]]:
    e_int_base = magnetic_energy_power(device)
    if auto_scale_heating:
        scale = device_heating_scale(device)
        nbi_peak = nbi_peak * scale
        p_rf_baseline = p_rf_baseline * scale

    def source(i: int) -> Tuple[float, float]:
        e_int = e_int_base * (1.0 + 0.05 * math.sin(i / 17.0))
        e_ext = nbi_wave_power(i, pulse_period, ramp, nbi_peak, p_rf_baseline)
        return e_int, e_ext

    return source


def device_Ef(device: DeviceParams, reference_device: DeviceParams = EAST) -> float:
    return Ef_REF * (magnetic_energy_power(device) / magnetic_energy_power(reference_device))


class ReferenceTable:
    def __init__(self, fb_lattice, Ef: float = Ef_REF):
        self.fb = fb_lattice
        self.Ef = Ef
        self.E_internal, self.E_external = self._build()
        self._mu_int, self._sd_int = self._robust_stats(self.E_internal)
        self._mu_ext, self._sd_ext = self._robust_stats(self.E_external)

    def _build(self):
        """Formules exactes : E_int = Ef*(fb1-fb0)/(fb1+fb0), E_ext = 2*Ef*fb0/(fb1+fb0)"""
        E_int, E_ext = [], []
        for pos in range(len(self.fb) - 1):
            fb0, fb1 = self.fb[pos], self.fb[pos + 1]
            denom = fb1 + fb0
            if denom == 0:
                E_int.append(float("nan"))
                E_ext.append(float("nan"))
                continue
            E_int.append(self.Ef * (fb1 - fb0) / denom)
            E_ext.append(2 * self.Ef * fb0 / denom)
        return E_int, E_ext

    @staticmethod
    def _robust_stats(values):
        finite = sorted(v for v in values if math.isfinite(v))
        if not finite:
            return 0.0, 1.0
        n = len(finite)
        lo, hi = int(0.01 * n), int(0.99 * n)
        trimmed = finite[lo:hi] if hi > lo else finite
        mu = sum(trimmed) / len(trimmed)
        var = sum((x - mu) ** 2 for x in trimmed) / max(1, len(trimmed) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        return mu, sd

    def normalize(self, e_int, e_ext) -> Tuple[float, float]:
        return (e_int - self._mu_int) / self._sd_int, (e_ext - self._mu_ext) / self._sd_ext


class AssociationIndex:
    def __init__(self, ref: ReferenceTable, d_seuil: float = 1.0):
        self.ref = ref
        self.d_seuil = d_seuil
        # CORRECTIF : la géométrie de recherche est construite à partir de
        # {|E_internal[i]|, E_external[i]}, jamais de E_internal signé.
        # Rappel de la règle : le signe négatif de E_internal dans la table
        # de référence n'indique pas une énergie négative (Ec = (1/2)m(vx²+vy²+vz²) >= 0
        # est toujours positive) -- c'est une étiquette indiquant qu'il faut
        # associer/chercher une triangularité négative, mais avec les valeurs
        # d'énergie {|E_internal[i]|, E_external[i]}. Le fait qu'un point modèle
        # soit éligible ou non à cette étiquette (E_internal < 0) est décidé
        # séparément par le filtre allow_negative dans query()/sorted_expectation() ;
        # ici, on ne fait que garantir que la distance géométrique elle-même
        # est calculée sur des grandeurs d'énergie toujours positives.
        z_int = np.array([ref.normalize(abs(e_i), e_e)[0]
                           for e_i, e_e in zip(ref.E_internal, ref.E_external)])
        z_ext = np.array([ref.normalize(abs(e_i), e_e)[1]
                           for e_i, e_e in zip(ref.E_internal, ref.E_external)])
        finite_mask = np.isfinite(z_int) & np.isfinite(z_ext)
        self._j_index = np.where(finite_mask)[0]
        self._z_int = z_int[finite_mask]
        self._z_ext = z_ext[finite_mask]

    def query(self, e_int_obs: float, e_ext_obs: float, allow_negative: bool = True):
        """
        INTERDIT l'appariement à des points modèle E_internal < 0 si
        allow_negative=False. Règle d'éligibilité de la triangularité négative :
        un point modèle à E_internal < 0 ne représente jamais une énergie
        négative -- c'est une étiquette de triangularité négative -- et il ne
        peut être utilisé pour l'appariement que si la triangularité négative
        est effectivement observée dans les données physiques.

        Conséquence géométrique : la comparaison se fait toujours dans l'espace
        {|E_internal|, E_external} (jamais sur E_internal signé), afin qu'un
        point physique (toujours positif) puisse correctement se rapprocher
        d'un point modèle à triangularité négative en utilisant ses valeurs
        d'énergie {|E_internal[i]|, E_external[i]} -- seule son éligibilité
        (via allow_negative) dépend du signe.
        """
        if len(self._j_index) == 0:
            return {"j_star": None, "D1": float("inf"), "D2": float("inf"),
                    "delta_D": 0.0, "N": 0}

        z_i_obs, z_e_obs = self.ref.normalize(abs(e_int_obs), e_ext_obs)
        d = np.hypot(self._z_int - z_i_obs, self._z_ext - z_e_obs)

        # Filtre optionnel : exclure les E_internal < 0 du modèle
        if not allow_negative:
            valid_mask = np.array([self.ref.E_internal[int(j)] >= 0
                                   for j in self._j_index])
            d = d.copy()
            d[~valid_mask] = np.inf

        if len(d) >= 2 and np.isfinite(d).sum() >= 2:
            finite_idx = np.where(np.isfinite(d))[0]
            if len(finite_idx) >= 2:
                idx2 = finite_idx[np.argpartition(d[finite_idx], 1)[:2]]
                idx2 = idx2[np.argsort(d[idx2])]
                best_i, second_i = idx2[0], idx2[1]
                best_d, second_d = float(d[best_i]), float(d[second_i])
                best_j = int(self._j_index[best_i])
            else:
                best_d = float("inf")
                second_d = float("inf")
                best_j = None
        else:
            if np.isfinite(d).any():
                best_i = int(np.argmin(d))
                best_d = float(d[best_i])
                second_d = float("inf")
                best_j = int(self._j_index[best_i])
            else:
                best_d = float("inf")
                second_d = float("inf")
                best_j = None

        n_within = int(np.count_nonzero(d <= self.d_seuil))

        return {
            "j_star": best_j,
            "D1": best_d,
            "D2": second_d,
            "delta_D": (second_d - best_d) if math.isfinite(second_d) else 0.0,
            "N": n_within,
        }

    def sorted_expectation(self, e_int_obs: float, e_ext_obs: float,
                           allow_negative: bool = True, k_neighbors: int = 50):
        """
        Diagnostic sort_mismatch conservé (v4), MAIS avec le même filtre
        d'éligibilité de la triangularité négative que query(). Ne déclenche
        jamais de disruption -- purement informatif.
        """
        # CORRECTIF (même règle que dans query()) : comparaison dans l'espace
        # {|E_internal|, E_external}, car E_internal négatif est une étiquette
        # de triangularité et non une énergie négative.
        z_i_obs, z_e_obs = self.ref.normalize(abs(e_int_obs), e_ext_obs)
        d = np.hypot(self._z_int - z_i_obs, self._z_ext - z_e_obs)

        if not allow_negative:
            valid_mask = np.array([self.ref.E_internal[int(j)] >= 0
                                   for j in self._j_index])
            d = d.copy()
            d[~valid_mask] = np.inf

        within = np.where(d <= self.d_seuil)[0]
        if len(within) < 3:
            k = min(k_neighbors, len(d))
            within = np.argpartition(d, k - 1)[:k]

        if len(within) == 0:
            return float("nan"), 0

        j_neigh = self._j_index[within]
        e_int_neigh = np.array([self.ref.E_internal[int(j)] for j in j_neigh])
        e_ext_neigh = np.array([self.ref.E_external[int(j)] for j in j_neigh])

        order = np.argsort(e_int_neigh)
        e_int_sorted = e_int_neigh[order]
        e_ext_sorted = e_ext_neigh[order]

        rank = int(np.searchsorted(e_int_sorted, e_int_obs))
        rank = min(max(rank, 0), len(e_ext_sorted) - 1)

        return float(e_ext_sorted[rank]), len(j_neigh)


def ef_from_physical(e_int_phys: float, e_ext_phys: float,
                     ef_prev: Optional[float] = None, alpha: float = 0.1) -> float:
    """
    Ef dynamique dérivé de la mesure physique (EWMA).
    raw = |E_internal,phys| + |E_external,phys|  (valeurs absolues pour garder
    le budget d'énergie scalaire positif, conformément au fait que l'énergie
    magnétique ne peut pas être négative).
    """
    raw = abs(e_int_phys) + abs(e_ext_phys)
    if ef_prev is None:
        return raw
    return (1 - alpha) * ef_prev + alpha * raw


def triangularity_from_model(e_internal_model: float, scale: float = Ef_REF) -> float:
    """Convention utilisateur conservée : étiquette diagnostique pure."""
    if e_internal_model < 0:
        return -math.tanh(abs(e_internal_model) / scale)
    return 0.0


def velocity_sign_convention(e_internal_model: float) -> Tuple[int, int, int]:
    """Convention utilisateur conservée : étiquette diagnostique pure."""
    sign_yz = -1 if e_internal_model < 0 else 1
    return (1, sign_yz, sign_yz)


class RealtimeFeed:
    def __init__(self, source: Callable[[int], Tuple[float, float]]):
        self._source = source
        self._manual_overrides: Dict[int, List[Optional[float]]] = {}

    def push_manual_correction(self, i: int, e_internal: Optional[float] = None,
                                e_external: Optional[float] = None):
        cur = self._manual_overrides.get(i, [None, None])
        if e_internal is not None:
            cur[0] = e_internal
        if e_external is not None:
            cur[1] = e_external
        self._manual_overrides[i] = cur

    def get(self, i: int) -> Tuple[float, float]:
        e_int, e_ext = self._source(i)
        if i in self._manual_overrides:
            m_int, m_ext = self._manual_overrides[i]
            if m_int is not None:
                e_int = m_int
            if m_ext is not None:
                e_ext = m_ext
        return e_int, e_ext


class NBISequencingEnvV5:
    """
    Version finale intégrant :
    - Ef dynamique (EWMA) dérivé de la mesure physique
    - sort_mismatch conservé comme diagnostic (jamais disruptif)
    - Règle d'éligibilité de la triangularité négative pour l'appariement :
      un point modèle à E_internal < 0 est une étiquette de triangularité
      négative, jamais une énergie négative ; il n'est utilisable pour
      l'appariement que si la triangularité négative est effectivement
      observée côté physique, auquel cas l'appariement utilise les valeurs
      d'énergie {|E_internal[i]|, E_external[i]}
    - Règle : si pas de triangularité négative physique observée,
      ne JAMAIS apparier à des points modèle E_internal < 0
    """
    ACTIONS = [1, 2, 3, 5, 8]

    def __init__(self, fb_lattice, feed: RealtimeFeed, device_id: int,
                 base_Ef: float = Ef_REF, rE: float = 2.0, d_seuil: float = 1.0,
                 use_ratio_guard: bool = True, use_association_guard: bool = False,
                 alpha_ef: float = 0.1, max_i: int = 2000,
                 separation_neg_pos: bool = True):
        self.fb_lattice = fb_lattice
        self.feed = feed
        self.device_id = device_id
        self.rE = rE
        self.d_seuil = d_seuil
        self.alpha_ef = alpha_ef
        self.max_i = max_i
        # NOUVEAU : applique la règle d'éligibilité de la triangularité négative.
        # Si True (défaut), les points modèle à E_internal < 0 ne sont jamais
        # utilisés pour l'appariement, SAUF si la source physique indique
        # explicitement une triangularité négative observée -- auquel cas
        # l'appariement se fait avec les valeurs d'énergie {|E_internal[i]|,
        # E_external[i]}, jamais avec une énergie considérée comme négative.
        self.separation_neg_pos = separation_neg_pos

        # Ef dynamique : initialisé à partir de la première mesure ou base device
        self.current_Ef = base_Ef
        self.ref = ReferenceTable(fb_lattice, Ef=self.current_Ef)
        self.assoc = AssociationIndex(self.ref, d_seuil=d_seuil)

        self.use_ratio_guard = use_ratio_guard
        self.use_association_guard = use_association_guard
        self.i = 0
        self._last_j = None

    def _is_negative_triangularity_observed(self, e_int_phys, e_ext_phys) -> bool:
        """
        Détermine si la triangularité négative est effectivement OBSERVÉE
        dans les données physiques. Par défaut : NON -- car la triangularité
        négative est un effet géométrique (bobines poloïdales) qui n'apparaît
        que si spécifiquement demandé. Le signe de E_internal,phys n'est PAS
        un indicateur de triangularité (l'énergie physique ne peut pas être
        négative : Ec >= 0 toujours). C'est cette méthode, et elle seule, qui
        décide si les points modèle à E_internal < 0 (étiquette de
        triangularité négative) deviennent éligibles à l'appariement.
        """
        return False  # Par défaut : pas de triangularité négative observée

    def reset(self):
        self.i = random.randint(0, self.max_i // 4)
        self._last_j = None
        # Recalage initial de Ef sur l'état courant
        e_int, e_ext = self.feed.get(self.i)
        if math.isfinite(e_int) and math.isfinite(e_ext):
            self.current_Ef = ef_from_physical(e_int, e_ext, None, self.alpha_ef)
            self.ref = ReferenceTable(self.fb_lattice, Ef=self.current_Ef)
            self.assoc = AssociationIndex(self.ref, d_seuil=self.d_seuil)
        return self._state()

    def _state(self):
        e_int, e_ext = self.feed.get(self.i)

        # Mise à jour dynamique de Ef(i) par EWMA basée sur la mesure physique
        if math.isfinite(e_int) and math.isfinite(e_ext):
            prev_ef = self.current_Ef
            self.current_Ef = ef_from_physical(e_int, e_ext, prev_ef, self.alpha_ef)
            if abs(self.current_Ef - prev_ef) > 1e-9:
                self.ref = ReferenceTable(self.fb_lattice, Ef=self.current_Ef)
                self.assoc = AssociationIndex(self.ref, d_seuil=self.d_seuil)

        # Déterminer si la triangularité négative est observée dans les données
        # physiques. Si séparation active ET pas de triangularité négative observée,
        # alors interdire l'appariement aux E_internal < 0 du modèle.
        allow_negative = not self.separation_neg_pos
        if self.separation_neg_pos:
            allow_negative = self._is_negative_triangularity_observed(e_int, e_ext)

        q = self.assoc.query(e_int, e_ext, allow_negative=allow_negative)
        ratio = abs(e_int / e_ext) if math.isfinite(e_ext) and abs(e_ext) > 1e-9 else 99.0
        delta_j = 0
        if q["j_star"] is not None and self._last_j is not None:
            delta_j = q["j_star"] - self._last_j
        return (
            round(min(ratio, 20.0), 1),
            round(min(q["D1"], 10.0), 1),
            round(min(q["delta_D"], 5.0), 1),
            min(q["N"], 20),
            max(-50, min(50, delta_j)) // 5,
            self.device_id,
        )

    def step(self, action_idx):
        jump = self.ACTIONS[action_idx]
        new_i = min(self.i + jump, self.max_i)
        self.i = new_i

        state = self._state()  # Met à jour Ef, table, et calcule l'état
        e_int, e_ext = self.feed.get(self.i)

        info = {}
        if not all(math.isfinite(x) for x in (e_int, e_ext)) or abs(e_ext) < 1e-9:
            return state, -10.0, True, {"reason": "etat_non_physique"}

        ratio = abs(e_int / e_ext)

        # Déterminer si la triangularité négative est observée
        allow_negative = not self.separation_neg_pos
        if self.separation_neg_pos:
            allow_negative = self._is_negative_triangularity_observed(e_int, e_ext)

        q = self.assoc.query(e_int, e_ext, allow_negative=allow_negative)
        info.update(q)
        info["ratio"] = ratio
        info["current_Ef"] = self.current_Ef
        info["separation_active"] = self.separation_neg_pos
        info["allow_negative"] = allow_negative

        # Diagnostic sort_mismatch conservé (PUREMENT INFORMATIF, jamais disruptif)
        e_ext_pred, n_neigh = self.assoc.sorted_expectation(
            e_int, e_ext, allow_negative=allow_negative
        )
        if math.isfinite(e_ext_pred):
            sort_mismatch = abs(e_ext - e_ext_pred) / (abs(e_ext) + 1e-9)
        else:
            sort_mismatch = float("nan")
        info["e_ext_predicted_by_sort"] = e_ext_pred
        info["sort_mismatch"] = sort_mismatch
        info["n_neigh"] = n_neigh

        # Étiquettes diagnostiques triangularité/vitesse (toujours purement informatives)
        if q["j_star"] is not None:
            e_int_model_matched = self.ref.E_internal[q["j_star"]]
            info["triangularity"] = triangularity_from_model(e_int_model_matched, scale=self.ref.Ef)
            info["velocity_signs"] = velocity_sign_convention(e_int_model_matched)
        else:
            info["triangularity"] = None
            info["velocity_signs"] = None

        # Garde-fous : sort_mismatch N'EST PAS un garde-fou ici
        disruption, reasons = False, []
        if self.use_ratio_guard and ratio > self.rE:
            disruption = True
            reasons.append(f"ratio_phys>{self.rE}")
        if self.use_association_guard and q["D1"] > self.assoc.d_seuil:
            disruption = True
            reasons.append("hors_modele")
        if q["j_star"] is None:
            disruption = True
            reasons.append("aucun_appariement_positif")

        self._last_j = q["j_star"] if q["j_star"] is not None else self._last_j

        if disruption:
            return state, -10.0, True, {"reason": "+".join(reasons), **info}

        reward = 1.0 - abs(ratio - 1.0)
        if q["j_star"] is not None:
            confidence_bonus = max(0.0, 1.0 - q["D1"]) * 0.05
            reward += confidence_bonus

        done = self.i >= self.max_i
        return state, reward, done, info


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.2, gamma=0.9, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        self.q = {}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def _q(self, state):
        return self.q.setdefault(state, [0.0] * self.n_actions)

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        qs = self._q(state)
        return max(range(self.n_actions), key=lambda i: qs[i])

    def update(self, state, action, reward, next_state, done):
        qs = self._q(state)
        next_max = 0.0 if done else max(self._q(next_state))
        qs[action] += self.alpha * (reward + self.gamma * next_max - qs[action])

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_multi_source(envs: List[NBISequencingEnvV5], n_episodes_per_device=400, max_steps=50):
    agent = QLearningAgent(n_actions=len(envs[0].ACTIONS))
    history = {env.device_id: [] for env in envs}
    total_rounds = n_episodes_per_device
    for _round in range(total_rounds):
        for env in envs:
            state = env.reset()
            steps, done = 0, False
            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
            agent.decay()
            history[env.device_id].append((steps, done, info.get("reason")))
    return agent, history


def evaluate(env: NBISequencingEnvV5, agent: QLearningAgent, n_episodes=30, max_steps=50):
    agent.eps = 0.0
    d_ratio = d_assoc = d_nonphys = d_no_match = completions = 0
    for _ in range(n_episodes):
        state = env.reset()
        steps, done, info = 0, False, {}
        while not done and steps < max_steps:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            steps += 1
        reason = info.get("reason", "")
        if done and reason:
            if "ratio" in reason:
                d_ratio += 1
            if "hors_modele" in reason:
                d_assoc += 1
            if "non_physique" in reason:
                d_nonphys += 1
            if "aucun_appariement_positif" in reason:
                d_no_match += 1
        elif steps >= max_steps or env.i >= env.max_i:
            completions += 1
    total_disruptions = sum([d_ratio, d_assoc, d_nonphys, d_no_match])
    assert total_disruptions + completions == n_episodes, (
        f"incohérence : {total_disruptions} disruptions + {completions} completions "
        f"!= {n_episodes} épisodes évalués"
    )
    return d_ratio, d_assoc, d_nonphys, d_no_match, completions, total_disruptions




if __name__ == "__main__":
    # Remplace : N_B = 6 ; fb_lattice = build_fb_lattice(s2_sample[0], n_b=N_B)
    fb_lattice = build_fb_lattice_from_chain()
    print(f"fb-lattice : {len(fb_lattice)} valeurs "
          f"(issu de la chaîne intervalle [{INTERVAL_INDEX}], chaîne #{CHAIN_RANK}, "
          f"chain_n_max={CHAIN_N_MAX})")

    # Vérification de l'identité algébrique
    print("\n--- Vérification de fidélité des formules ---")
    ref_test = ReferenceTable(fb_lattice, Ef=Ef_REF)
    max_err = max(abs(ei + ee - Ef_REF)
                  for ei, ee in zip(ref_test.E_internal, ref_test.E_external)
                  if math.isfinite(ei) and math.isfinite(ee))
    n_neg = sum(1 for ei in ref_test.E_internal if math.isfinite(ei) and ei < 0)
    n_pos = sum(1 for ei in ref_test.E_internal if math.isfinite(ei) and ei >= 0)
    print(f"  Identité E_int + E_ext = Ef : erreur max = {max_err:.2e}")
    print(f"  Lignes E_internal < 0 : {n_neg} ({100*n_neg/len(fb_lattice):.1f}%)")
    print(f"  Lignes E_internal >= 0: {n_pos} ({100*n_pos/len(fb_lattice):.1f}%)")

    devices = [EAST, W7X, DIIID]
    envs = []
    print("\n--- Configurations dispositifs ---")
    for idx, dev in enumerate(devices):
        Ef_dev = device_Ef(dev)
        scale = device_heating_scale(dev)
        feed = RealtimeFeed(make_device_source(dev))
        env = NBISequencingEnvV5(
            fb_lattice=fb_lattice, feed=feed, device_id=idx,
            base_Ef=Ef_dev, rE=2.0, d_seuil=1.0,
            use_ratio_guard=True, use_association_guard=False, alpha_ef=0.1,
            separation_neg_pos=True  # Séparation stricte activée par défaut
        )
        envs.append(env)
        print(f"  {dev.name:15s} | Base Ef_device={Ef_dev:6.3f} MW | heating_scale={scale:.3f}")

    print("\n--- Entraînement multi-sources avec Ef dynamique + séparation négatif/positif ---")
    agent, history = train_multi_source(envs, n_episodes_per_device=400)

    print("\n--- Évaluation (par dispositif, politique gloutonne) ---")
    print(f" {'Dispositif':15s} | {'ratio':>5s} {'assoc':>5s} {'nonphys':>7s} {'no_match':>8s} | "
          f"{'TOTAL DISRUPTIONS':>18s} | {'completions':>11s}")
    for env, dev in zip(envs, devices):
        d_ratio, d_assoc, d_nonphys, d_no_match, completions, total = evaluate(env, agent, n_episodes=30)
        print(f" {dev.name:15s} | {d_ratio:5d} {d_assoc:5d} {d_nonphys:7d} {d_no_match:8d} | "
              f" {total:18d} | {completions:9d}/30")

    # Diagnostic sort_mismatch sur trajectoire naïve (purement informatif)
    print("\n--- Diagnostic sort_mismatch sur trajectoire naïve (garde-fous OFF) ---")
    probe_env = NBISequencingEnvV5(
        fb_lattice=fb_lattice, feed=RealtimeFeed(make_device_source(EAST)),
        device_id=0, rE=2.0, d_seuil=1.0,
        use_ratio_guard=False, use_association_guard=False, alpha_ef=0.1,
        separation_neg_pos=True)
    ratios, D1s, sort_mismatches = [], [], []
    probe_env.reset()
    for i in range(300):
        _, _, _, info = probe_env.step(0)  # action=0 -> increment de 1
        if "ratio" in info:
            ratios.append(info["ratio"])
            D1s.append(info["D1"])
            sort_mismatches.append(info["sort_mismatch"])

    print(f"  sort_mismatch : min={min(sort_mismatches):.3f} "
          f"max={max(sort_mismatches):.3f} "
          f"moyenne={sum(sort_mismatches)/len(sort_mismatches):.3f}")