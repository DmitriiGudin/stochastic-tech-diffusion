from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize

from scipy.optimize import minimize
from pyproj import Transformer

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


def poisson_deviance(obs: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    pred = np.maximum(pred, eps)

    term = np.zeros_like(obs, dtype=float)
    positive = obs > 0
    term[positive] = obs[positive] * np.log(obs[positive] / pred[positive])
    return float(2.0 * np.sum(term - (obs - pred)))


def mesh_lonlat(data, epsg_project: int = 5070):
    pts_m = data.mesh_points_km * 1000.0
    tr = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(pts_m[:, 0], pts_m[:, 1])
    return np.asarray(lon), np.asarray(lat)


def plot_node_values(
    ax,
    *,
    data,
    values: np.ndarray,
    title: str,
    epsg_project: int,
    vmin: float | None = None,
    vmax: float | None = None,
    scale: str = "log1p",
    n_layers: int = 10,
):
    lon, lat = mesh_lonlat(data, epsg_project=epsg_project)
    tri = data.triangles

    raw_vals = np.asarray(values, dtype=float)

    if scale == "log1p":
        vals = np.log1p(np.clip(raw_vals, 0.0, None))
    elif scale == "linear":
        vals = np.clip(raw_vals, 0.0, None)
    else:
        raise ValueError("scale must be 'linear' or 'log1p'.")

    finite = np.isfinite(vals)

    if vmin is None:
        vmin = float(np.nanmin(vals[finite])) if np.any(finite) else 0.0
    if vmax is None:
        vmax = float(np.nanquantile(vals[finite], 0.99)) if np.any(finite) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    triang = mtri.Triangulation(lon, lat, tri)
    ax.triplot(triang, linewidth=0.2, color="0.82", alpha=0.65, zorder=1)

    s = max(5.0, 5000.0 / max(np.sqrt(len(vals)), 1.0)) * 0.35

    finite_vals = vals[finite]
    if finite_vals.size == 0:
        layer_edges = np.array([vmin, vmax])
    else:
        qs = np.linspace(0.0, 1.0, int(n_layers) + 1)
        layer_edges = np.nanquantile(finite_vals, qs)
        layer_edges = np.unique(layer_edges)
        if layer_edges.size < 2:
            layer_edges = np.array([vmin, vmax])

    mappable = None

    for ell in range(layer_edges.size - 1):
        lo = layer_edges[ell]
        hi = layer_edges[ell + 1]

        if ell == layer_edges.size - 2:
            mask = finite & (vals >= lo) & (vals <= hi)
        else:
            mask = finite & (vals >= lo) & (vals < hi)

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        # Critical fix: sort within layer from smaller to larger.
        idx = idx[np.argsort(vals[idx])]

        mappable = ax.scatter(
            lon[idx],
            lat[idx],
            c=vals[idx],
            s=s,
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.55 + 0.45 * (ell + 1) / max(layer_edges.size - 1, 1),
            zorder=2 + ell,
        )

    if mappable is None:
        mappable = ax.scatter(
            lon,
            lat,
            c=np.zeros_like(vals),
            s=s,
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.55,
            zorder=2,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    return mappable


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
    
    deviance_sssb = poisson_deviance(annual_obs, annual_model)
    deviance_bass = poisson_deviance(annual_obs, annual_bass)
    
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
    
    field_values = {
        "U": details["cum_mu_U"],
        "V": details["cum_mu_V"],
        "I": details["final_I"],
        "J": details["final_J"],
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.ravel()
    
    for ax, (name, vals) in zip(axes, field_values.items()):
        log_vals = np.log1p(np.clip(vals, 0.0, None))
    
        vmin_i = 0.0
        vmax_i = float(np.nanquantile(log_vals, 0.99))
        if vmax_i <= vmin_i:
            vmax_i = 1.0
    
        sc = plot_node_values(
            ax,
            data=data,
            values=vals,
            title=f"{name} at end of data period",
            epsg_project=int(cfg_named["mesh"]["epsg_project"]),
            vmin=vmin_i,
            vmax=vmax_i,
        )
    
        fig.colorbar(
            sc,
            ax=ax,
            fraction=0.045,
            pad=0.02,
            label=f"ln(1 + {name})",
        )
    
    fig_path_fields = out_dir / "final_fields_UV_IJ_log1p.png"
    fig.savefig(fig_path_fields, dpi=200)
    plt.close(fig)
    
    actual_cum_node = data.Y.sum(axis=0)
    pred_cum_node = details["cum_mu_total"]
    
    adoption_scale = str(cfg_named["density"].get("adoption_plot_scale", "log1p"))
    
    if adoption_scale == "log1p":
        actual_vals_for_scale = np.log1p(actual_cum_node)
        pred_vals_for_scale = np.log1p(pred_cum_node)
        actual_cbar_label = "ln(1 + observed cumulative count)"
        pred_cbar_label = "ln(1 + predicted cumulative mean)"
    else:
        actual_vals_for_scale = actual_cum_node
        pred_vals_for_scale = pred_cum_node
        actual_cbar_label = "Observed cumulative count"
        pred_cbar_label = "Predicted cumulative mean"
    
    actual_vmin = 0.0
    actual_vmax = float(np.nanquantile(actual_vals_for_scale, 0.99))
    if actual_vmax <= actual_vmin:
        actual_vmax = 1.0
    
    pred_vmin = 0.0
    pred_vmax = float(np.nanquantile(pred_vals_for_scale, 0.99))
    if pred_vmax <= pred_vmin:
        pred_vmax = 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    sc0 = plot_node_values(
        axes[0],
        data=data,
        values=actual_cum_node,
        title="Observed cumulative adoptions",
        epsg_project=int(cfg_named["mesh"]["epsg_project"]),
        vmin=actual_vmin,
        vmax=actual_vmax,
        scale=adoption_scale,
    )
    
    fig.colorbar(
        sc0,
        ax=axes[0],
        fraction=0.045,
        pad=0.02,
        label=actual_cbar_label,
    )
    
    sc1 = plot_node_values(
        axes[1],
        data=data,
        values=pred_cum_node,
        title="Predicted cumulative mean counts",
        epsg_project=int(cfg_named["mesh"]["epsg_project"]),
        vmin=pred_vmin,
        vmax=pred_vmax,
        scale=adoption_scale,
    )
    
    fig.colorbar(
        sc1,
        ax=axes[1],
        fraction=0.045,
        pad=0.02,
        label=pred_cbar_label,
    )
    
    fig_path_spatial = out_dir / "observed_vs_predicted_cumulative_log1p.png"
    fig.savefig(fig_path_spatial, dpi=200)
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
        "poisson_deviance": {
            "sssb": deviance_sssb,
            "bass": deviance_bass,
        },
        "plots": {
            "cumulative_linear": str(fig_path),
            "cumulative_log1p": str(fig_path_log),
            "final_fields_log1p": str(fig_path_fields),
            "observed_vs_predicted_spatial_log1p": str(fig_path_spatial),
        },
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[METRICS]")
    print(json.dumps(metrics, indent=2))
    print("[PLOT] wrote:", fig_path)
    print("[PLOT] wrote:", fig_path_log)
    print("[PLOT] wrote:", fig_path_fields)
    print("[PLOT] wrote:", fig_path_spatial)
    print("[DATA] wrote:", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())