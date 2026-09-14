"""
Algorithme_A_decomp_vf2_2.py  (autonome -- aucun import local)
=================================================================

Version autonome : contient desormais, dans ce seul fichier,
  1) l'equation caracteristique et la resolution de s2(t) par brentq
     (voir "Algorithme 2" fourni par l'utilisateur),
  2) le calcul de TOUS les b(t,k) = (asin(signe*|s2(t)|) - 2*k*pi)/ln(2)
     (fonction generate_candidates),
  3) le chainage vf2_2 (recherche successive de (N1,b1),(N2,b2),...),
  4) la decomposition COMPLETE de chaque (N1,b1) puis (N2,b2) en Nf
     triplets {t_k,sign_k,k_k,s2_val_k,b_k},
  5) la construction du vecteur composite b[1..M] par concatenation
     (N1,b1) -> (N2,b2) -> (N1,b1) -> (N2,b2) -> ... des etapes de la
     chaine,
  6) la logique physique d'origine d'Algorithme_A (Ef=5, fb = un
     ECHANTILLON Monte-Carlo de paires de b[] -- voir note echelle
     ci-dessous --, puis pour chaque paire consecutive de l'echantillon
     (fb[t2],fb[t2+1]) : Einternal, Eexternal, comptage id1/id2/id3/
     total) -- physique CONSERVEE A L'IDENTIQUE.

Remplace le calcul original b[l] = (asin(s2[t]) - 2*k_l*pi)/log(2) pour
des combinaisons fixes (k_1<...<k_4) : b[] provient maintenant d'UNE
chaine reelle et de sa decomposition complete (voir generate_candidates
et build_composite_b ci-dessous).

NOTE ECHELLE / ECHANTILLONNAGE : CHAIN_N_MAX = N_MAX = 65536 = 2**16
(echelle REELLE de N1/N2, pas une valeur reduite pour test). A cette
echelle, une chaine de plusieurs etapes donne un vecteur composite b[]
de plusieurs centaines de milliers d'elements, et calculer TOUTES ses
differences par paires (O(M^2)) est infaisable (des dizaines de
milliards de paires). fb est donc construit par sample_fb_pairs() :
un echantillon Monte-Carlo de PAIR_SAMPLE_SIZE paires (i1,i2) tirees
uniformement dans b[] (i1 != i2), au lieu de l'enumeration exhaustive
de l'original -- resultat statistiquement representatif, pas exact.
"""
import random as _random
import numpy as np
from math import asin, log, pi
from fractions import Fraction as Frac
from bisect import bisect_left
from scipy.optimize import brentq



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
# 1. CHAINAGE -- GROUPE 2 : "COUPLAGE ENTRE ETAPES CONSECUTIVES"
#    Etape 1 : recherche LIBRE de (N1,N2). Etape n>=2 : b1_n = b2_(n-1),
#    N1_n != N2_(n-1), N1_n + N2_(n-1) <= N_MAX (recherche vectorisee
#    numpy sur N2 in [1,n2_max]).
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


def best_rational_constrained(ratio, n1_max, n2_max, forbidden_n1):
    """Bornes asymetriques 1<=N1<=n1_max, 1<=N2<=n2_max, N1 != forbidden_n1
    -- recherche vectorisee directe sur N2 in [1,n2_max]."""
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
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        if n1_max is None:
            fr = best_rational(ratio, n_max)
        else:
            fr = best_rational_constrained(ratio, n1_max, n_max, forbidden_n1)
        if fr is None:
            continue
        N1, N2 = fr
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
            if prev_N2 is None:
                res = best_partner_fast(current_b, current_sign, target, fp, n_max)
            else:
                n1_max = n_max - prev_N2
                res = best_partner_fast(current_b, current_sign, target, fp, n_max,
                                         n1_max=n1_max, forbidden_n1=prev_N2)
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
# Parametres -- t_max/k_range repris des fichiers de chainage d'origine ;
# CHAIN_N_MAX = N_MAX = 65536 = echelle REELLE (voir note ci-dessus)
# ---------------------------------------------------------------------------
CHAIN_T_MAX, CHAIN_K_RANGE = 2000, 20   # pool de RECHERCHE DE CHAINE (repris des fichiers de chainage d'origine)
CHAIN_N_MAX = N_MAX                     # = 65536 = 2**16 -- echelle reelle de N1/N2
INTERVAL_INDEX = 1                      # intervalle [X0_LIST[i],X0_LIST[i+1]] (index 0 est structurellement vide)
CHAIN_RANK = 0                          # 0 = premiere chaine trouvee sur cet intervalle

DECOMP_T_MAX, DECOMP_K_RANGE = 2000, 20 # pool DEDIE a la decomposition (repris des fichiers de chainage d'origine)
DECOMP_ITERS = 300000                   # iterations du recuit simule (valeur des fichiers de reference)

PAIR_SAMPLE_SIZE = 2_000_000            # nombre de paires (i1,i2) echantillonnees dans b[] pour former fb
                                         # (remplace l'enumeration EXHAUSTIVE O(M^2), infaisable a l'echelle N_MAX)

Ef = 5


def build_b_from_chain():
    print("1. Génération du pool de recherche de chaîne...", flush=True)
    candidates = generate_candidates(t_max=CHAIN_T_MAX, k_range=CHAIN_K_RANGE, verbose=False)
    fp = FastPool(candidates)

    print(f"2. Recherche de chaîne sur l'intervalle [{INTERVAL_INDEX}] de X0_LIST "
          f"(n_max={CHAIN_N_MAX})...", flush=True)
    x0_1, x0_2, remaining, anchors, chains = run_chain(
        INTERVAL_INDEX, X0_LIST, candidates, fp, n_max=CHAIN_N_MAX, max_chains=CHAIN_RANK + 1
    )
    if not chains:
        raise RuntimeError(f"Aucune chaîne trouvée sur l'intervalle [{INTERVAL_INDEX}] ({x0_1}, {x0_2}).")
    chain_record = chains[CHAIN_RANK]
    chain = chain_record["chain"]
    print(f"   -> chaîne #{CHAIN_RANK} : {chain_record['n_l']} étape(s) sur {len(remaining)} cible(s).", flush=True)

    print("3. Génération du pool de décomposition...", flush=True)
    decomp_pool = generate_decomposition_pool(t_max=DECOMP_T_MAX, k_range=DECOMP_K_RANGE)

    print("4. Décomposition COMPLÈTE de chaque (N1,b1) puis (N2,b2) de la chaîne...", flush=True)
    annotate_chain_with_decompositions(chain, decomp_pool, iters=DECOMP_ITERS, seed=0)

    b_list = build_composite_b(chain)
    print(f"   -> vecteur composite b[] : {len(b_list):,} valeurs "
          f"(remplace les b[1..4] de l'Algorithme_A d'origine)", flush=True)
    return b_list


if __name__ == "__main__":
    b0 = build_b_from_chain()
    M = len(b0)

    id1 = id2 = id3 = 0
    total = 0

    n_pairs_possible = M * (M - 1) // 2
    print(f"5. Échantillonnage de {PAIR_SAMPLE_SIZE:,} paires parmi "
          f"{n_pairs_possible:,} paires possibles...", flush=True)
    fb = sample_fb_pairs(b0, PAIR_SAMPLE_SIZE, seed=1)

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

    print("Number of Einternal < 0:", id1)
    print("Number of Eexternal < 0:", id2)
    print("Number of Einternal > Eexternal:", id3)
    print("Total rows:", total)
