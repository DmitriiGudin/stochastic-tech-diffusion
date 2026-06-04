from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import gammaln
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SSSBFitParams:
    p: float
    q: float
    gamma_J: float
    k_J: float
    D: float
    S0: float
    r0: float
    r1: float = 0.0
    r2: float = 0.0


@dataclass(frozen=True)
class SSSBFitConfig:
    dt_years: float = 0.05
    eps_mu: float = 1e-12
    use_covariates: bool = True
    normalize_nll: bool = True


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def standardize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)

    out = np.zeros_like(x, dtype=float)
    if not np.any(good):
        return out

    mu = float(np.nanmean(x[good]))
    sd = float(np.nanstd(x[good]))

    if sd <= 0:
        return out

    out[good] = (x[good] - mu) / sd
    return out


def build_capacity(
    *,
    population: np.ndarray,
    pv_potential: np.ndarray,
    transmission_distance_km: np.ndarray,
    params: SSSBFitParams,
    use_covariates: bool,
) -> np.ndarray:
    population = np.asarray(population, dtype=float)
    population = np.clip(population, 0.0, None)

    eta = np.full_like(population, float(params.r0), dtype=float)

    if use_covariates:
        z_pv = standardize_feature(pv_potential)
        z_grid = standardize_feature(transmission_distance_km)

        eta += float(params.r1) * z_pv
        eta -= float(params.r2) * z_grid

    r = softplus(eta)
    return population * r


def observed_driven_nll(
    *,
    Y: np.ndarray,
    years: np.ndarray,
    population: np.ndarray,
    pv_potential: np.ndarray,
    transmission_distance_km: np.ndarray,
    L: csr_matrix,
    params: SSSBFitParams,
    cfg: SSSBFitConfig,
    return_details: bool = False,
):
    """
    Vectorized observation-driven Poisson NLL.

    Vectorized over nodes.
    Looped over years and substeps.
    """
    Y = np.asarray(Y, dtype=float)
    years = np.asarray(years, dtype=int)

    if Y.ndim != 2:
        raise ValueError("Y must be shaped (n_years, n_nodes).")

    n_years, n_nodes = Y.shape

    if n_years == 0:
        return 0.0 if not return_details else (0.0, {})

    K = build_capacity(
        population=population,
        pv_potential=pv_potential,
        transmission_distance_km=transmission_distance_km,
        params=params,
        use_covariates=cfg.use_covariates,
    )

    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    D = float(params.D)
    S0 = float(params.S0)

    if p < 0 or q < 0 or gamma_J < 0 or k_J < 0 or D < 0 or S0 < 0:
        if return_details:
            return np.inf, {}
        return np.inf

    dt_req = float(cfg.dt_years)
    n_sub = int(round(1.0 / dt_req))
    if n_sub < 1:
        raise ValueError("dt_years is too large.")

    dt = 1.0 / n_sub

    I = np.zeros(n_nodes, dtype=float)
    J = np.zeros(n_nodes, dtype=float)
    W_cum = np.zeros(n_nodes, dtype=float)

    mu = np.zeros_like(Y, dtype=float)

    for yi in range(n_years):
        Y_year = Y[yi]

        R = np.clip(K - W_cum, 0.0, None)
        jump = Y_year / n_sub

        for _ in range(n_sub):
            info_effect = I / (1.0 + I)
            hazard = p + q * info_effect

            mu[yi] += R * hazard * dt

            J_plus = J + jump

            I = I + dt * gamma_J * J_plus

            LJ = L @ J_plus
            J = J_plus + dt * (-k_J * J_plus + D * LJ + S0)

            I = np.maximum(I, 0.0)
            J = np.maximum(J, 0.0)

        W_cum += Y_year

    eps = float(cfg.eps_mu)
    nll_terms = mu - Y * np.log(mu + eps) + gammaln(Y + 1.0)
    nll = float(np.sum(nll_terms))

    if cfg.normalize_nll:
        denom = max(float(np.sum(Y)), 1.0)
        nll /= denom

    if not return_details:
        return nll

    details = {
        "mu": mu,
        "capacity": K,
        "final_I": I,
        "final_J": J,
        "final_cumulative_observed": W_cum,
        "years": years,
        "dt_years_used": np.array([dt]),
        "n_substeps_per_year": np.array([n_sub]),
    }

    return nll, details