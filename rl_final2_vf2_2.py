"""
rl_final2_vf2_2.py -- rl_extended_sketch réécrit, GROUPE "vf2_2"
(moteur de chaînage : couplage N1/N2 croisé, recherche vectorisée numpy)

Seule build_fb_lattice() est remplacée : au lieu des C(pool,n_b) combinaisons
de k, on chaîne successivement à travers les k=1..n_b+1 (un pas de chaînage
vf2_2 par transition k_i -> k_(i+1)), on décompose (N1,b1) PUIS (N2,b2) à
chaque pas via decompose_bf_into_bk, on concatène le tout, et fb_lattice est
pris comme les différences consécutives de cette liste concaténée. Voir
la partie commune intégrée ci-dessous pour le détail complet (NOTE SIGNE / NOTE TAILLE).
Le reste du fichier (ReferenceTable, AssociationIndex, l'environnement RL,
l'entraînement Q-learning) est inchangé.
"""

# ============================================================================
# PARTIE COMMUNE INTÉGRÉE DIRECTEMENT (ex-module _common_decomp_chain.py)
# t_max / k_range fixés comme dans les fichiers de chaînage d'origine
# (algorithme_chainage_beta_vf2_1/2_avec_decomposition (vf1-2).py) :
# generate_candidates(t_max=2000, k_range=20), pool unique recherche+décomposition.
# ============================================================================
import math
import random as _random
import time
from math import asin, log, pi
from fractions import Fraction as Frac
from bisect import bisect_left

import numpy as np
from scipy.optimize import brentq

LN2 = log(2)
N_MAX = 65536

# --- Plafonds pratiques (ajustables) pour garder le calcul exécutable ------
SEARCH_T_MAX = 2000      # pool de RECHERCHE de partenaire (t=1..SEARCH_T_MAX)
SEARCH_K_RANGE = 20
DECOMP_T_MAX = 2000      # pool de DÉCOMPOSITION (plus grand, cf. fichiers sources)
DECOMP_K_RANGE = 20
DECOMP_ITERS = 20000    # recuit simulé par décomposition (300000 dans les
                        # fichiers sources ; réduit ici par défaut pour rester
                        # exécutable sur 100..200 valeurs de t -- à augmenter
                        # si besoin de décompositions plus précises)
N1_N2_SOFT_CAP = 65536  # ~2**16 (= N_MAX) : on ne retaille plus quasiment
                        # jamais N1/N2 avant décomposition (contre 4000
                        # auparavant). ATTENTION COÛT : decompose_bf_into_bk
                        # sur Nf~65000 prend ~1.5-1.8s (au lieu de ~0.06s pour
                        # Nf=4000) avec DECOMP_ITERS=20000 -- un t complet
                        # (N_B pas x 2 décompositions) peut donc prendre
                        # jusqu'à ~15s pour Algorithme_A (N_B=4) et ~40s pour
                        # Algorithme_B (N_B=12, x2 signes), soit potentiellement
                        # plusieurs dizaines de minutes pour les 100 valeurs de
                        # t. Réduisez ce plafond (ou DECOMP_ITERS) si besoin
                        # d'un calcul plus rapide.


# ============================================================================
# 1. GÉNÉRATION DU POOL DE CANDIDATS -- résolution de l'équation
#    caractéristique GÉNÉRALE (Algorithme 2), valable pour tout t >= 1 :
#
#      0 = |8s2^6-24s2^4+24s2^2-8|^(2/3) * (24k3^2 s2^6-72k3^2 s2^4-24k3^2)
#          + 4 s2^4,   avec k3 = -t
#
#    (t positif ici correspond à l'indice -t dans Algorithme_A/B, même
#    convention qu'avant -- seule la FORMULE et la résolution changent :
#    il n'y a plus de table de constantes figées, s2(t) est calculé à la
#    demande par brentq, pour tout t >= 1, avec mémoïsation.)
# ============================================================================
from functools import lru_cache

def _equation(s2, k3):
    A = 8 * s2 ** 6 - 24 * s2 ** 4 + 24 * s2 ** 2 - 8
    return abs(A) ** (2 / 3) * (24 * k3 ** 2 * s2 ** 6 - 72 * k3 ** 2 * s2 ** 4 - 24 * k3 ** 2) + 4 * s2 ** 4

# NOTE BRACKET : pour k3=-t avec t petit (jusqu'a ~590), la racine reste dans
# le bracket fixe [-0.9999,-0.0001] proposé initialement. Mais pour t plus
# grand (on va jusqu'à t_max=2000 ici, cf. SEARCH_T_MAX/DECOMP_T_MAX plus
# bas), la racine se rapproche de -1 (ex. t=2000 -> s2 ~ -0.99997) et sort de
# ce bracket fixe, ce qui fait échouer brentq ("f(a) et f(b) doivent avoir
# des signes différents"). On cherche donc D'ABORD, par un balayage fin de
# (-1,0), l'intervalle où l'équation change de signe, PUIS on appelle brentq
# sur ce bracket -- la formule de l'équation reste rigoureusement identique.
_S2_SCAN_N = 2000

@lru_cache(maxsize=None)
def get_s2_chain(t):
    """Résout numériquement l'équation caractéristique pour k3 = -t et
    renvoie la racine s2(t) dans (-1, 0), pour tout t >= 1 entier. C'est
    ici -- et seulement ici -- que la résolution a lieu (mémoïsée avec
    lru_cache pour éviter de la refaire à chaque appel)."""
    k3 = -t
    lo, hi = -0.999999999999, -1e-9
    step = (hi - lo) / (_S2_SCAN_N - 1)
    prev_x, prev_v = lo, _equation(lo, k3)
    for i in range(1, _S2_SCAN_N):
        x = lo + i * step
        v = _equation(x, k3)
        if prev_v == 0:
            return prev_x
        if (prev_v < 0) != (v < 0):
            return brentq(_equation, prev_x, x, args=(k3,), xtol=1e-14)
        prev_x, prev_v = x, v
    raise RuntimeError(f"get_s2_chain: aucune racine trouvée pour t={t}")

def get_s2_for_index(t_neg):
    """Convenience : t_neg est l'indice négatif utilisé dans Algorithme_A/B
    (ex. t_neg=-18) ; renvoie s2 correspondant, via get_s2_chain(-t_neg)
    (même convention k3=-t que ci-dessus, donc k3=t_neg)."""
    return get_s2_chain(-t_neg)

def generate_candidates(t_max, k_range, verbose=False):
    cands = []
    for t in range(1, t_max + 1):
        s2 = get_s2_chain(t)
        for sign in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k != 0:
                    b = (asin(sign * abs(s2)) - 2 * k * pi) / LN2
                    cands.append((t, k, sign, b))
    if verbose:
        print(f"  [generate_candidates] {len(cands)} candidats "
              f"(t_max={t_max}, k_range={k_range})", flush=True)
    return cands


class FastPool:
    """Pré-indexe le pool (trié par signe) pour une recherche par dichotomie."""
    def __init__(self, candidates):
        self.pos = sorted([(b, t, k) for (t, k, sign, b) in candidates if sign == 1])
        self.neg = sorted([(b, t, k) for (t, k, sign, b) in candidates if sign == -1])
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


def build_pools(search_t_max=SEARCH_T_MAX, search_k_range=SEARCH_K_RANGE,
                 decomp_t_max=DECOMP_T_MAX, decomp_k_range=DECOMP_K_RANGE, verbose=True):
    if verbose:
        print("[pools] génération du pool de recherche...", flush=True)
    search_pool = generate_candidates(search_t_max, search_k_range, verbose)
    fp = FastPool(search_pool)
    if verbose:
        print("[pools] génération du pool de décomposition...", flush=True)
    decomp_pool = generate_candidates(decomp_t_max, decomp_k_range, verbose)
    return search_pool, fp, decomp_pool


# ============================================================================
# 2. MOTEUR "vf2_1" -- b hérité, poids N1 contraint par rapport au N2 précédent
#    (repris de algorithme_chainage_beta_vf2_1_avec_decomposition)
# ============================================================================
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
        denom = (anchor_b - target_x0)
        if denom == 0:
            continue
        ratio = (target_x0 - b2) / denom
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
        denom = (anchor_b - target_x0)
        if denom == 0:
            continue
        ratio = (target_x0 - b2) / denom
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


# ============================================================================
# 3. MOTEUR "vf2_2" -- couplage N1/N2 croisé (recherche vectorisée numpy)
#    (repris de algorithme_chainage_beta_vf2_2_avec_decomposition)
# ============================================================================
def best_rational_constrained_np(ratio, n1_max, n2_max, forbidden_n1):
    if ratio <= 0 or n1_max < 1:
        return None
    q = np.arange(1, n2_max + 1)
    p = np.round(ratio * q).astype(np.int64)
    valid = (p >= 1) & (p <= n1_max)
    if forbidden_n1 is not None:
        valid &= (p != forbidden_n1)
    if not valid.any():
        return None
    p_v, q_v = p[valid], q[valid]
    err_ratio = np.abs(p_v / q_v - ratio)
    idx = np.argmin(err_ratio)
    return int(p_v[idx]), int(q_v[idx])


def best_partner_fast(anchor_b, anchor_sign, target_x0, fp, n_max=N_MAX, k_near=10,
                       n1_max=None, forbidden_n1=None):
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)
    best = None
    for (t2, k2, b2) in cand_set:
        denom = (anchor_b - target_x0)
        if denom == 0:
            continue
        ratio = (target_x0 - b2) / denom
        if n1_max is None:
            fr = best_rational(ratio, n_max)
        else:
            fr = best_rational_constrained_np(ratio, n1_max, n_max, forbidden_n1)
        if fr is None:
            continue
        N1, N2 = fr
        xc = (N1 * anchor_b + N2 * b2) / (N1 + N2)
        err = abs(xc - target_x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign_needed, b2, N1, N2, xc)
    return best


# ============================================================================
# 4. UN PAS DE CHAÎNAGE, quel que soit le moteur ("vf2_1" ou "vf2_2")
# ============================================================================
def get_pair_step(engine, anchor_b, anchor_sign, target_x0, prev_N2, fp, n_max=N_MAX, k_near=10):
    """Retourne (err,t2,k2,sign2,b2,N1,N2,xc) ou None si aucun partenaire
    n'a été trouvé. `engine` in {"vf2_1", "vf2_2"}."""
    if engine == "vf2_1":
        if prev_N2 is None:
            return best_partner_free(anchor_b, anchor_sign, target_x0, fp, n_max, k_near)
        return best_partner_constrained(anchor_b, anchor_sign, target_x0, prev_N2, fp, n_max, k_near)
    elif engine == "vf2_2":
        if prev_N2 is None:
            return best_partner_fast(anchor_b, anchor_sign, target_x0, fp, n_max, k_near)
        n1_max = n_max - prev_N2
        return best_partner_fast(anchor_b, anchor_sign, target_x0, fp, n_max, k_near,
                                  n1_max=n1_max, forbidden_n1=prev_N2)
    else:
        raise ValueError(f"engine inconnu : {engine!r} (attendu 'vf2_1' ou 'vf2_2')")


# ============================================================================
# 5. DÉCOMPOSITION Nf.b_f = sum(b_k) (reprise à l'identique des fichiers de
#    chaînage) + concaténation (N1,b1) puis (N2,b2) -> tableau b étendu
# ============================================================================
def decompose_bf_into_bk(Nf, b_f_target, decomp_pool, iters=DECOMP_ITERS, seed=0, exclude=None):
    _random.seed(seed)
    n = len(decomp_pool)
    exclude = exclude or set()
    pool_idx = [i for i in range(n) if i not in exclude]
    assert len(pool_idx) >= Nf, (
        f"pas assez de candidats ({len(pool_idx)}) pour Nf={Nf} -- "
        f"augmentez DECOMP_T_MAX/DECOMP_K_RANGE"
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
        if new_cost < cur_cost or _random.random() < math.e ** (-(new_cost - cur_cost) / max(T, 1e-12)):
            idx[pos] = trial
            idx_set.discard(old_i); idx_set.add(trial)
            current_sum = new_sum
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost, best_idx = cur_cost, list(idx)

    bk_list = [decomp_pool[i][3] for i in best_idx]
    return best_cost, bk_list


def _cap(n, cap=N1_N2_SOFT_CAP):
    """Retaille N vers `cap` s'il le dépasse (voir NOTE TAILLE en tête de
    fichier) -- garde le rapport de la décomposition exécutable à chaque t."""
    return min(n, cap) if cap else n


def build_extended_b(engine, anchor_b, anchor_sign, target_x0, prev_N2, fp, decomp_pool,
                      n_max=N_MAX, k_near=10, decomp_iters=DECOMP_ITERS, seed=0,
                      n1_n2_cap=N1_N2_SOFT_CAP):
    """Exécute UN pas de chaînage (`engine`), puis décompose (N1,b1) PUIS
    (N2,b2), et renvoie :
        b_extended : liste = b_k(N1,b1) concaténée avec b_k(N2,b2)
        N1, N2     : tailles effectivement utilisées (après plafonnement)
        step       : le tuple brut (err,t2,k2,sign2,b2,N1,N2,xc) ou None
    """
    step = get_pair_step(engine, anchor_b, anchor_sign, target_x0, prev_N2, fp, n_max, k_near)
    if step is None:
        return None, None, None, None
    err, t2, k2, sign2, b2, N1_raw, N2_raw, xc = step
    N1 = max(1, _cap(N1_raw, n1_n2_cap))
    N2 = max(1, _cap(N2_raw, n1_n2_cap))

    _, bk1 = decompose_bf_into_bk(N1, anchor_b, decomp_pool, iters=decomp_iters, seed=seed)
    _, bk2 = decompose_bf_into_bk(N2, b2, decomp_pool, iters=decomp_iters, seed=seed + 1)

    b_extended = bk1 + bk2
    return b_extended, N1, N2, step


# ============================================================================
# 6. VARIANTE POUR rl_extended_sketch_v5_final(2).py : ici il n'y a qu'UNE
#    seule valeur s2_t (pas de boucle sur t), et l'ancienne combinatoire
#    portait sur n_b valeurs de k. On remplace cette combinatoire par UNE
#    CHAÎNE successive à travers les n_b k (k=1 -> cible k=2 -> cible k=3 ->
#    ...), chaque pas produisant (N1,b1) puis (N2,b2), chacun décomposé et
#    concaténé -- "puis (N1,b1) puis (N2,b2), puis (N1,b1) puis (N2,b2), ..."
#    exactement dans l'esprit de la demande. `sign_list` = [-1] reproduit le
#    v5_final d'origine (une seule convention de signe) ; [1, -1] reproduit
#    le v5_final2 d'origine (les deux signes, concaténés).
# ============================================================================
def _k_pool(n_b, k_lo=None, k_hi=None):
    if k_lo is None or k_hi is None:
        k_lo, k_hi = -n_b, n_b
    return [k for k in range(k_lo, k_hi + 1) if k != 0][:max(2, n_b + 1)]


def chain_decompose_over_ks(engine, s2_t, sign, n_b, fp, decomp_pool,
                             n_max=N_MAX, decomp_iters=DECOMP_ITERS,
                             n1_n2_cap=N1_N2_SOFT_CAP, seed=0):
    """Chaîne successivement à travers n_b+1 ancres b(k_i) = (asin(sign*abs(s2_t))
    - 2*k_i*pi)/ln2 (k_1=1..k_(n_b+1) -- écart ~2*pi/ln2~9.06 entre ancres
    consécutives, du même ordre de grandeur que les écarts entre cibles
    X0_LIST des fichiers algorithme_chainage_beta_vf2_1/2.py, ~0.1 à ~20 --
    contrairement à un chaînage sur des t consécutifs, où l'écart ~1e-6 est
    beaucoup trop petit et produit des N1/N2 non représentatifs, cf. NOTE
    ÉCHELLE ci-dessous), et renvoie la CONCATÉNATION des décompositions de
    chaque (N1,b1) puis (N2,b2) rencontrés, ainsi que la liste des (N1,N2)
    par étape (pour diagnostic/affichage)."""
    ks = list(range(1, n_b + 2))  # n_b+1 ancres -> n_b pas de chaînage
    b_full = []
    steps_info = []
    prev_N2 = None
    for i in range(len(ks) - 1):
        b_anchor = (asin(sign * abs(s2_t)) - 2 * ks[i] * pi) / LN2
        b_target = (asin(sign * abs(s2_t)) - 2 * ks[i + 1] * pi) / LN2
        b_ext, N1, N2, step = build_extended_b(
            engine, b_anchor, sign, b_target, prev_N2, fp, decomp_pool,
            n_max=n_max, decomp_iters=decomp_iters, seed=seed + i, n1_n2_cap=n1_n2_cap,
        )
        if b_ext is None:
            break
        b_full.extend(b_ext)
        steps_info.append((N1, N2))
        prev_N2 = N2
    return b_full, steps_info


def build_fb_lattice_decomp(engine, s2_t, n_b, fp, decomp_pool, sign_list=(-1,),
                             n_max=N_MAX, decomp_iters=DECOMP_ITERS,
                             n1_n2_cap=N1_N2_SOFT_CAP, seed=0):
    """Remplace build_fb_lattice() des fichiers rl_extended_sketch : au lieu
    des C(pool,n_b) combinaisons de k, chaîne successivement à travers les
    k=1..n_b+1 (une fois par signe de `sign_list`), décompose chaque
    (N1,b1)/(N2,b2), concatène le tout, puis prend les différences
    consécutives comme fb_lattice (cf. NOTE TAILLE en tête de fichier)."""
    b_full = []
    for s_idx, sign in enumerate(sign_list):
        b_step, _ = chain_decompose_over_ks(
            engine, s2_t, sign, n_b, fp, decomp_pool,
            n_max=n_max, decomp_iters=decomp_iters, n1_n2_cap=n1_n2_cap,
            seed=seed + 1000 * s_idx,
        )
        b_full.extend(b_step)
    return consecutive_fb(b_full)


def consecutive_fb(b_list):
    """fb[i] = b[i] - b[i+1] -- voir NOTE TAILLE en tête de fichier (on ne
    reprend pas toutes les C(N,2) paires du code original, impraticable
    dès que N1+N2 dépasse quelques milliers)."""
    return [b_list[i] - b_list[i + 1] for i in range(len(b_list) - 1)]

# ============ FIN DE LA PARTIE COMMUNE INTÉGRÉE ============

ENGINE = "vf2_2"


import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict

random.seed(0)

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

# s2_sample résolu via l'équation caractéristique générale (Algorithme 2)
# -- mêmes indices t=-18..-1 que dans le fichier d'origine, plus de
# constantes figées.
s2_sample = [get_s2_for_index(t) for t in range(-18, 0)]


def auto_k_range(n_b: int) -> Tuple[int, int]:
    """TOUJOURS [-n_b, n_b] comme demandé."""
    if n_b < 1:
        raise ValueError("N_b doit être un entier >= 1")
    return -n_b, n_b


def estimate_combinations(n_b: int, k_lo: Optional[int] = None, k_hi: Optional[int] = None) -> int:
    if k_lo is None or k_hi is None:
        k_lo, k_hi = auto_k_range(n_b)
    pool_size = (k_hi - k_lo + 1) - 1  # exclure k=0
    if pool_size < n_b:
        return 0
    return math.comb(pool_size, n_b)


def build_fb_lattice(s2_t, n_b=4, k_lo=None, k_hi=None, max_combos=2_000_000,
                      fp=None, decomp_pool=None):
    """Remplacée : chaîne successive vf2_1 à travers k=1..n_b+1, puis
    décomposition (N1,b1)/(N2,b2) à chaque pas, concaténation, et fb_lattice
    = différences consécutives de la liste concaténée (voir en-tête de
    fichier et _common_decomp_chain.py)."""
    if fp is None or decomp_pool is None:
        raise ValueError("build_fb_lattice (version décomposition) requiert fp et decomp_pool "
                          "(cf. build_pools() dans _common_decomp_chain.py)")
    return build_fb_lattice_decomp(ENGINE, s2_t, n_b, fp, decomp_pool,
                                    sign_list=(1, -1), decomp_iters=DECOMP_ITERS)


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
    N_B = 6
    print("0. Génération des pools (recherche + décomposition)...", flush=True)
    _, fp, decomp_pool = build_pools(SEARCH_T_MAX, SEARCH_K_RANGE, DECOMP_T_MAX, DECOMP_K_RANGE)
    # Toujours utiliser [-n_b, n_b] pour les k chaînés
    fb_lattice = build_fb_lattice(s2_sample[0], n_b=N_B, fp=fp, decomp_pool=decomp_pool)
    print(f"fb-lattice : {len(fb_lattice)} valeurs (N_b={N_B}, plage k=[-{N_B},{N_B}])")

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
