from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from configs import DEFAULT, CONFIGS
from fit_data_utils import build_sssb_fit_data
from sssb_solver import SSSBFitParams, SSSBFitConfig, observed_driven_nll


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(name: str) -> dict:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name}. Valid configs: {sorted(CONFIGS)}")
    return deep_update(DEFAULT, CONFIGS[name])


def bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if p <= 0 or q <= 0 or M <= 0:
        return np.full_like(t, np.nan, dtype=float)

    e = np.exp(-(p + q) * t)
    return M * (1.0 - e) / (1.0 + (q / p) * e)


def fit_bass_curve(t: np.ndarray, cum_obs: np.ndarray) -> dict:
    final_obs = float(cum_obs[-1])

    def unpack(theta):
        p = np.exp(theta[0])
        q = np.exp(theta[1])
        M = final_obs + np.exp(theta[2])
        return p, q, M

    def obj(theta):
        p, q, M = unpack(theta)
        pred = bass_cumulative(t, p, q, M)
        return float(np.mean((pred - cum_obs) ** 2))

    theta0 = np.array([np.log(0.01), np.log(0.3), np.log(max(final_obs, 1.0))])
    res = minimize(obj, theta0, method="Nelder-Mead", options={"maxiter": 5000})

    p, q, M = unpack(res.x)
    return {
        "p": float(p),
        "q": float(q),
        "M": float(M),
        "success": bool(res.success),
        "rmse": float(np.sqrt(obj(res.x))),
    }


def model_monthly_expected_counts(
    *,
    data,
    params: SSSBFitParams,
    cfg: SSSBFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      month_t       shape (n_years*12 + 1,)
      cum_model     cumulative expected total adoptions at monthly grid
    """
    Y = np.asarray(data.Y, dtype=float)
    n_years, n_nodes = Y.shape

    _, details = observed_driven_nll(
        Y=data.Y,
        years=data.years,
        population=data.population,
        pv_potential=data.pv_potential,
        transmission_distance_km=data.transmission_distance_km,
        L=data.L,
        params=params,
        cfg=cfg,
        return_details=True,
    )

    annual_mu = details["mu"].sum(axis=1)

    # Monthly interpolation of model annual expected counts.
    # This matches the data convention: annual counts are spread uniformly.
    monthly_counts = np.repeat(annual_mu / 12.0, 12)
    cum_model = np.concatenate([[0.0], np.cumsum(monthly_counts)])

    month_t = np.arange(cum_model.size, dtype=float) / 12.0
    return month_t, cum_model


def observed_monthly_curve(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    annual_counts = Y.sum(axis=1)
    monthly_counts = np.repeat(annual_counts / 12.0, 12)
    cum = np.concatenate([[0.0], np.cumsum(monthly_counts)])
    t = np.arange(cum.size, dtype=float) / 12.0
    return t, cum


def yearly_rmse_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def yearly_rmse_cumulative(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.cumsum(obs) - np.cumsum(pred)) ** 2)))


def yearly_rmse_log1p_counts(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(obs) - np.log1p(pred)) ** 2)))


def yearly_rmse_log1p_cumulative(obs: np.ndarray, pred: np.ndarray) -> float:
    obs_cum = np.cumsum(obs)
    pred_cum = np.cumsum(pred)
    return float(np.sqrt(np.mean((np.log1p(obs_cum) - np.log1p(pred_cum)) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--fit_json", default=None, type=str)
    args = parser.parse_args()

    cfg_named = load_config(args.config)

    fit_json = Path(args.fit_json) if args.fit_json else Path("out") / args.config / "fit_result.json"
    if not fit_json.exists():
        raise FileNotFoundError(fit_json)

    with open(fit_json, "r", encoding="utf-8") as f:
        fit_payload = json.load(f)

    mesh_path = Path(fit_payload["data"]["mesh"])
    features_path = Path(fit_payload["data"]["features"])
    lspv_csv = Path(fit_payload["data"]["lspv_csv"])

    data = build_sssb_fit_data(
        msh_path=mesh_path,
        node_features_npz=features_path,
        lspv_csv=lspv_csv,
        epsg_project=int(cfg_named["mesh"]["epsg_project"]),
        population_key=str(cfg_named["fit"]["population_key"]),
    )

    params = SSSBFitParams(**fit_payload["params"])
    solver_cfg = SSSBFitConfig(**fit_payload["solver_config"])

    nll, details = observed_driven_nll(
        Y=data.Y,
        years=data.years,
        population=data.population,
        pv_potential=data.pv_potential,
        transmission_distance_km=data.transmission_distance_km,
        L=data.L,
        params=params,
        cfg=solver_cfg,
        return_details=True,
    )

    annual_obs = data.Y.sum(axis=1)
    annual_model = details["mu"].sum(axis=1)

    rmse_yearly = yearly_rmse_counts(annual_obs, annual_model)
    rmse_cum = yearly_rmse_cumulative(annual_obs, annual_model)

    t_obs_month, cum_obs_month = observed_monthly_curve(data.Y)
    t_model_month, cum_model_month = model_monthly_expected_counts(
        data=data,
        params=params,
        cfg=solver_cfg,
    )

    # Bass fit to monthly cumulative observed curve.
    bass = fit_bass_curve(t_obs_month, cum_obs_month)
    cum_bass = bass_cumulative(t_obs_month, bass["p"], bass["q"], bass["M"])
    
    # Convert monthly Bass cumulative curve to annual increments.
    # month index 12*k corresponds to end of year k.
    bass_cum_year_end = cum_bass[12::12]
    bass_cum_year_start = np.concatenate([[0.0], bass_cum_year_end[:-1]])
    annual_bass = bass_cum_year_end - bass_cum_year_start
    
    rmse_yearly_log1p = yearly_rmse_log1p_counts(annual_obs, annual_model)
    rmse_cum_log1p = yearly_rmse_log1p_cumulative(annual_obs, annual_model)
    
    rmse_yearly_bass = yearly_rmse_counts(annual_obs, annual_bass)
    rmse_cum_bass = yearly_rmse_cumulative(annual_obs, annual_bass)
    
    rmse_yearly_log1p_bass = yearly_rmse_log1p_counts(annual_obs, annual_bass)
    rmse_cum_log1p_bass = yearly_rmse_log1p_cumulative(annual_obs, annual_bass)

    out_dir = Path("out") / args.config / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    calendar_months = data.years[0] + t_obs_month

    ax.plot(calendar_months, cum_obs_month, label="Observed data")
    ax.plot(calendar_months, cum_bass, label="Classic Bass fit")
    ax.plot(calendar_months, cum_model_month, label="SSSB expected cumulative")

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative LSPV adoptions")
    ax.set_title("Cumulative adoption curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig_path = out_dir / "cumulative_adoption_curve.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(calendar_months, np.log1p(cum_obs_month), label="Observed data")
    ax.plot(calendar_months, np.log1p(cum_bass), label="Classic Bass fit")
    ax.plot(calendar_months, np.log1p(cum_model_month), label="SSSB expected cumulative")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("log1p cumulative LSPV adoptions")
    ax.set_title("Cumulative adoption curve, log1p scale")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig_path_log = out_dir / "cumulative_adoption_curve_log1p.png"
    fig.savefig(fig_path_log, dpi=200)
    plt.close(fig)

    metrics = {
        "config": args.config,
        "fit_json": str(fit_json),
        "nll": float(nll),
        "years": data.years.astype(int).tolist(),
        "observed_total": float(annual_obs.sum()),
        "model_expected_total": float(annual_model.sum()),
        "rmse_yearly_counts": {
            "sssb": rmse_yearly,
            "bass": rmse_yearly_bass,
        },
        "rmse_cumulative_counts": {
            "sssb": rmse_cum,
            "bass": rmse_cum_bass,
        },
        "rmse_yearly_log1p_counts": {
            "sssb": rmse_yearly_log1p,
            "bass": rmse_yearly_log1p_bass,
        },
        "rmse_cumulative_log1p_counts": {
            "sssb": rmse_cum_log1p,
            "bass": rmse_cum_log1p_bass,
        },
        "bass": bass,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[METRICS]")
    print(json.dumps(metrics, indent=2))
    print("[PLOT] wrote:", fig_path)
    print("[PLOT] wrote:", fig_path_log)
    print("[DATA] wrote:", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())