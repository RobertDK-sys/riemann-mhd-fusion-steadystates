"""
algorithme_A_vf2_2.py -- Algorithme_A réécrit, GROUPE "vf2_2"
(moteur de chaînage : couplage N1/N2 croisé, recherche vectorisée numpy)

Le module de chaînage/décomposition est intégré directement ci-dessous
(section "PARTIE COMMUNE INTÉGRÉE"). Détail de la substitution
b[1..N] -> décomposition en chaîne. Résumé : pour chaque t, on prend comme
ancre b(t), on chaîne UN pas vers la cible b(t+1) avec le moteur vf2_1, on
obtient (N1,b1=b(t)) et (N2,b2), puis on décompose CHACUN en b_k via
decompose_bf_into_bk. Le tableau b[1..N1+N2] (autrefois rempli par
b[1..4] = 4 valeurs à k fixés) est remplacé par la concaténation des deux
listes de b_k. fb est ensuite pris comme les différences consécutives de ce
tableau étendu (voir NOTE TAILLE dans le module commun).
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
Ef = 5
ANCHOR_SIGN = -1      # cf. NOTE SIGNE dans la partie commune intégrée
N_B = 4               # nombre de k chaînés (b[1..4] à l'origine, k=1..5)

id1 = id2 = id3 = total = 0

# s2(t) résolu via l'équation caractéristique générale (Algorithme 2),
# pour tout t : plus de table de constantes figées.
s2 = {t: get_s2_for_index(t) for t in range(-100, 0)}


if __name__ == "__main__":
    print("1. Génération des pools (recherche + décomposition)...", flush=True)
    _, fp, decomp_pool = build_pools(SEARCH_T_MAX, SEARCH_K_RANGE, DECOMP_T_MAX, DECOMP_K_RANGE)

    for t in range(-100, 0):
        # NOTE ÉCHELLE : on chaîne ici sur les k=1..N_B+1 pour ce t (écart
        # ~2*pi/ln2~9.06 entre ancres consécutives), et non plus sur des t
        # consécutifs (écart ~1e-6, beaucoup trop petit) -- voir NOTE ÉCHELLE
        # dans chain_decompose_over_ks ci-dessus. Cela rapproche les N1/N2
        # obtenus de ceux d'algorithme_chainage_beta_vf2_1/2.py (tous deux
        # potentiellement grands, dizaines de milliers), au lieu du motif
        # "N1 grand / N2 minuscule" observé avec le chaînage sur t.
        b_extended, steps_info = chain_decompose_over_ks(
            ENGINE, s2[t], ANCHOR_SIGN, N_B, fp, decomp_pool,
            decomp_iters=DECOMP_ITERS, seed=abs(t),
        )
        if not b_extended:
            print(f"t={t} : aucun partenaire de chaînage trouvé, t ignoré.")
            continue

        fb = consecutive_fb(b_extended)
        tmax = len(fb)

        for t2 in range(0, tmax - 1):
            fb0 = fb[t2]
            fb_next = fb[t2 + 1]
            denominator = fb_next + fb0
            if denominator != 0:
                Einternal = (Ef * fb_next - Ef * fb0) / denominator
                Eexternal = (2 * Ef * fb0) / denominator

                if Einternal < 0:
                    id1 += 1
                if Eexternal < 0:
                    id2 += 1
                if Einternal > Eexternal:
                    id3 += 1
                total += 1

        print(f"t={t} : {len(steps_info)} étape(s) N1/N2={steps_info} "
              f"-> {tmax} valeurs fb ({total} lignes cumulées)", flush=True)

    print("Number of Einternal < 0:", id1)
    print("Number of Eexternal < 0:", id2)
    print("Number of Einternal > Eexternal:", id3)
    print("Total rows:", total)
