from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from pyproj import Transformer
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


def mesh_lonlat(data, epsg_project: int = 5070):
    pts_m = data.mesh_points_km * 1000.0
    tr = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(pts_m[:, 0], pts_m[:, 1])
    return np.asarray(lon), np.asarray(lat)


def bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
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
    return {"p": float(p), "q": float(q), "M": float(M)}


def observed_monthly_curve(Y: np.ndarray):
    annual = Y.sum(axis=1)
    monthly = np.repeat(annual / 12.0, 12)
    cum = np.concatenate([[0.0], np.cumsum(monthly)])
    t = np.arange(cum.size, dtype=float) / 12.0
    return t, cum


def information_effect(I: np.ndarray, params: SSSBFitParams) -> np.ndarray:
    I = np.maximum(np.asarray(I, dtype=float), 0.0)
    a = float(getattr(params, "FI_a", 1.0))
    b = float(getattr(params, "FI_b", 1.0))
    c = float(getattr(params, "FI_c", 1e6))

    base = I / (1.0 + a * I)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.maximum(np.power(base, b) * (-np.expm1(-c * I)), 0.0)
    
    
def initialize_from_seed_year(
    *,
    data,
    years: np.ndarray,
    params: SSSBFitParams,
    solver_cfg: SSSBFitConfig,
    seed_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Condition on all observed years from years[0] through seed_year, inclusive.

    Returns:
      U, V, I, J at the end of seed_year,
      seed_idx,
      seed_cum_hist with shape (seed_idx + 1, n_nodes)
    """
    matches = np.where(years == int(seed_year))[0]
    if matches.size == 0:
        raise ValueError(f"seed_year={seed_year} not found in years.")

    seed_idx = int(matches[0])
    n_nodes = data.Y.shape[1]

    U = np.zeros(n_nodes, dtype=np.int64)
    V = np.zeros(n_nodes, dtype=np.int64)
    I = np.zeros(n_nodes, dtype=float)
    J = np.zeros(n_nodes, dtype=float)

    seed_cum_hist = np.zeros((seed_idx + 1, n_nodes), dtype=float)

    dt = float(solver_cfg.dt_years)
    n_sub = int(round(1.0 / dt))
    dt = 1.0 / n_sub

    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    D = float(params.D)
    S0 = float(params.S0)
    L = data.L

    for yi in range(seed_idx + 1):
        Y_seed = data.Y[yi].astype(float)

        # Treat all conditioned observed events as innovations.
        U += Y_seed.astype(np.int64)

        jump = Y_seed / n_sub

        for _ in range(n_sub):
            J_plus = J + jump
            I_new = I + dt * gamma_J * J_plus
            J_new = J_plus + dt * (-k_J * J_plus + D * (L @ J_plus) + S0)

            I = np.maximum(I_new, 0.0)
            J = np.maximum(J_new, 0.0)

        seed_cum_hist[yi] = U + V

    return U, V, I, J, seed_idx, seed_cum_hist


def sssb_one_step(
    *,
    rng,
    U,
    V,
    I,
    J,
    capacity,
    L,
    params,
    dt,
):
    R = np.maximum(capacity - U - V, 0.0)

    FI = information_effect(I, params)

    rate_U = float(params.p) * R
    rate_V = float(params.q) * FI * R
    rate_total = rate_U + rate_V

    lam = np.maximum(rate_total * dt, 0.0)
    d_total = rng.poisson(lam)

    cap_remaining = np.maximum(np.ceil(R).astype(int), 0)
    d_total = np.minimum(d_total, cap_remaining)

    innov_prob = np.ones_like(rate_total, dtype=float)
    np.divide(rate_U, rate_total, out=innov_prob, where=rate_total > 0.0)
    innov_prob = np.clip(innov_prob, 0.0, 1.0)

    dU = rng.binomial(d_total, innov_prob)
    dV = d_total - dU

    U = U + dU
    V = V + dV

    J_plus = J + d_total.astype(float)

    I_new = I + dt * float(params.gamma_J) * J_plus
    J_new = J_plus + dt * (
        -float(params.k_J) * J_plus
        + float(params.D) * (L @ J_plus)
        + float(params.S0)
    )

    I = np.maximum(I_new, 0.0)
    J = np.maximum(J_new, 0.0)

    return U, V, I, J, dU, dV


def simulate_sssb_stochastic(
    *,
    data,
    params: SSSBFitParams,
    solver_cfg: SSSBFitConfig,
    capacity: np.ndarray,
    years: np.ndarray,
    forecast_year: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    start_year = int(years[0])
    all_years = np.arange(start_year, int(forecast_year) + 1)

    n_nodes = capacity.size
    dt = float(solver_cfg.dt_years)
    n_sub = int(round(1.0 / dt))
    dt = 1.0 / n_sub

    cum_hist = np.zeros((all_years.size, n_nodes), dtype=float)
    
    if getattr(solver_cfg, "condition_on_seed_year", False):
        U, V, I, J, seed_idx, seed_cum_hist = initialize_from_seed_year(
            data=data,
            years=years,
            params=params,
            solver_cfg=solver_cfg,
            seed_year=int(solver_cfg.seed_year),
        )
    else:
        U = np.zeros(n_nodes, dtype=np.int64)
        V = np.zeros(n_nodes, dtype=np.int64)
        I = np.zeros(n_nodes, dtype=float)
        J = np.zeros(n_nodes, dtype=float)
        seed_idx = -1
        seed_cum_hist = None

    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    D = float(params.D)
    S0 = float(params.S0)
    L = data.L

    for yy, year in enumerate(all_years):
        if yy <= seed_idx:
            cum_hist[yy] = seed_cum_hist[yy]
            print(
                f"[SSSB sim] year={year} "
                f"total_cum={float(cum_hist[yy].sum()):.0f} "
                f"max_node_cum={float(cum_hist[yy].max()):.0f} "
                f"seed-conditioned"
            )
            continue
        
        year_new_adoptions = 0
        
        for _ in range(n_sub):
            U, V, I, J, dU, dV = sssb_one_step(
                rng=rng,
                U=U,
                V=V,
                I=I,
                J=J,
                capacity=capacity,
                L=L,
                params=params,
                dt=dt,
            )
            year_new_adoptions += int(np.sum(dU + dV))

        cum_hist[yy] = U + V
        
        year_total = float(cum_hist[yy].sum())
        year_max_node = float(cum_hist[yy].max())
        print(
            f"[SSSB sim] year={year} "
            f"new={year_new_adoptions:.0f} "
            f"total_cum={year_total:.0f} "
            f"max_node_cum={year_max_node:.0f} "
            f"max_I={float(np.max(I)):.3e} "
            f"max_J={float(np.max(J)):.3e}"
        )

    return {
        "years": all_years,
        "cum": cum_hist,
        "U": U,
        "V": V,
        "I": I,
        "J": J,
    }


def simulate_sssb_stochastic_batch(
    *,
    data,
    params: SSSBFitParams,
    solver_cfg: SSSBFitConfig,
    capacity: np.ndarray,
    years: np.ndarray,
    forecast_year: int,
    seed: int,
    n_sims: int,
) -> dict:
    rng = np.random.default_rng(seed)

    start_year = int(years[0])
    all_years = np.arange(start_year, int(forecast_year) + 1)

    n_nodes = capacity.size
    dt = float(solver_cfg.dt_years)
    n_sub = int(round(1.0 / dt))
    dt = 1.0 / n_sub

    capacity = np.asarray(capacity, dtype=np.float32)
    L_T = data.L.T.tocsr()

    cum_hist = np.zeros((n_sims, all_years.size, n_nodes), dtype=np.float32)

    if getattr(solver_cfg, "condition_on_seed_year", False):
        U0, V0, I0, J0, seed_idx, seed_cum_hist = initialize_from_seed_year(
            data=data,
            years=years,
            params=params,
            solver_cfg=solver_cfg,
            seed_year=int(solver_cfg.seed_year),
        )

        U = np.repeat(U0[None, :], n_sims, axis=0).astype(np.int32)
        V = np.repeat(V0[None, :], n_sims, axis=0).astype(np.int32)
        I = np.repeat(I0[None, :], n_sims, axis=0).astype(np.float32)
        J = np.repeat(J0[None, :], n_sims, axis=0).astype(np.float32)
    else:
        U = np.zeros((n_sims, n_nodes), dtype=np.int32)
        V = np.zeros((n_sims, n_nodes), dtype=np.int32)
        I = np.zeros((n_sims, n_nodes), dtype=np.float32)
        J = np.zeros((n_sims, n_nodes), dtype=np.float32)
        seed_idx = -1
        seed_cum_hist = None

    p = float(params.p)
    q = float(params.q)
    gamma_J = float(params.gamma_J)
    k_J = float(params.k_J)
    D = float(params.D)
    S0 = float(params.S0)

    for yy, year in enumerate(all_years):
        if yy <= seed_idx:
            cum_hist[:, yy, :] = seed_cum_hist[yy][None, :]
            continue

        for _ in range(n_sub):
            R = np.maximum(capacity[None, :] - U - V, 0.0)

            FI = information_effect(I, params)
            rate_U = p * R
            rate_V = q * FI * R
            rate_total = rate_U + rate_V

            lam = np.maximum(rate_total * dt, 0.0)
            d_total = rng.poisson(lam)

            cap_remaining = np.maximum(np.ceil(R).astype(np.int32), 0)
            d_total = np.minimum(d_total, cap_remaining)

            innov_prob = np.ones_like(rate_total, dtype=np.float32)
            np.divide(rate_U, rate_total, out=innov_prob, where=rate_total > 0.0)
            innov_prob = np.clip(innov_prob, 0.0, 1.0)

            dU = rng.binomial(d_total, innov_prob).astype(np.int32)
            dV = d_total.astype(np.int32) - dU

            U += dU
            V += dV

            J_plus = J + d_total.astype(np.float32)

            I_new = I + dt * gamma_J * J_plus
            LJ = J_plus @ L_T
            J_new = J_plus + dt * (-k_J * J_plus + D * LJ + S0)

            I = np.maximum(I_new, 0.0).astype(np.float32)
            J = np.maximum(J_new, 0.0).astype(np.float32)

        cum_hist[:, yy, :] = U + V

    final_totals = cum_hist[:, -1, :].sum(axis=1)

    return {
        "years": all_years,
        "cum_hist": cum_hist,
        "mean_cum": cum_hist.mean(axis=0),
        "std_cum": cum_hist.std(axis=0),
        "final_totals": final_totals,
    }


def simulate_bass_baseline(
    *,
    annual_expected: np.ndarray,
    weights: np.ndarray,
    years: np.ndarray,
    forecast_year: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    start_year = int(years[0])
    all_years = np.arange(start_year, int(forecast_year) + 1)

    n_nodes = weights.size
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(n_nodes, 1 / n_nodes)

    cum = np.zeros(n_nodes, dtype=int)
    hist = np.zeros((all_years.size, n_nodes), dtype=float)

    for k, mu_total in enumerate(annual_expected):
        total = rng.poisson(max(float(mu_total), 0.0))
        yearly = rng.multinomial(total, weights)
        cum += yearly
        hist[k] = cum

    return {"years": all_years, "cum": hist}


def simulate_bass_baseline_batch(
    *,
    annual_expected: np.ndarray,
    weights: np.ndarray,
    years: np.ndarray,
    forecast_year: int,
    seed: int,
    n_sims: int,
) -> dict:
    rng = np.random.default_rng(seed)

    start_year = int(years[0])
    all_years = np.arange(start_year, int(forecast_year) + 1)

    n_nodes = weights.size
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(n_nodes, 1.0 / n_nodes)

    cum = np.zeros((n_sims, n_nodes), dtype=np.int32)
    hist = np.zeros((n_sims, all_years.size, n_nodes), dtype=np.float32)

    for yy, mu_total in enumerate(annual_expected):
        totals = rng.poisson(max(float(mu_total), 0.0), size=n_sims)

        for rr in range(n_sims):
            yearly = rng.multinomial(int(totals[rr]), weights)
            cum[rr] += yearly.astype(np.int32)

        hist[:, yy, :] = cum

    return {
        "years": all_years,
        "cum_hist": hist,
        "mean_cum": hist.mean(axis=0),
        "std_cum": hist.std(axis=0),
        "final_totals": hist[:, -1, :].sum(axis=1),
    }


def triangle_boundary_edges(triangles: np.ndarray):
    counts = {}
    for a, b, c in triangles:
        for i, j in ((a, b), (b, c), (c, a)):
            e = tuple(sorted((int(i), int(j))))
            counts[e] = counts.get(e, 0) + 1
    return [e for e, c in counts.items() if c == 1]


def draw_node_map(
    ax,
    *,
    data,
    values: np.ndarray,
    title: str,
    epsg_project: int,
    vmin: float,
    vmax: float,
):
    lon, lat = mesh_lonlat(data, epsg_project=epsg_project)
    tri = data.triangles
    triang = mtri.Triangulation(lon, lat, tri)

    ax.triplot(triang, linewidth=0.2, color="0.75", alpha=0.5, zorder=1)

    # Boundary only, thick black.
    for i, j in triangle_boundary_edges(tri):
        ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="black", linewidth=1.1, zorder=2)

    vals = np.asarray(values, dtype=float)
    s = max(5.0, 5000.0 / max(np.sqrt(vals.size), 1.0)) * 0.35

    idx = np.argsort(vals)
    sc = ax.scatter(
        lon[idx],
        lat[idx],
        c=vals[idx],
        s=s,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    return sc


def save_static_panels(
    *,
    data,
    panels: list[tuple[str, np.ndarray]],
    out_path: Path,
    epsg_project: int,
    ncols: int,
    title: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_vals = np.concatenate([np.asarray(v, dtype=float).ravel() for _, v in panels])
    vmin = 0.0
    vmax = float(np.nanmax(all_vals))
    if vmax <= vmin:
        vmax = 1.0

    n = len(panels)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    last_sc = None
    for ax, (name, vals) in zip(axes, panels):
        last_sc = draw_node_map(
            ax,
            data=data,
            values=vals,
            title=name,
            epsg_project=epsg_project,
            vmin=vmin,
            vmax=vmax,
        )

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.colorbar(last_sc, ax=axes[:len(panels)].tolist(), fraction=0.035, pad=0.02, label="Cumulative count")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("[SIM plot] saved:", out_path)


def save_animation(
    *,
    data,
    panels_by_year: list[list[tuple[str, np.ndarray]]],
    years: np.ndarray,
    out_path: Path,
    epsg_project: int,
    ncols: int,
    title: str,
    fps: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_vals = []
    for panels in panels_by_year:
        for _, vals in panels:
            all_vals.append(np.asarray(vals, dtype=float).ravel())
    all_vals = np.concatenate(all_vals)

    vmin = 0.0
    vmax = float(np.nanmax(all_vals))
    if vmax <= vmin:
        vmax = 1.0

    n_panels = len(panels_by_year[0])
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    def update(frame):
        for ax in axes:
            ax.clear()

        panels = panels_by_year[frame]
        last_sc = None
        for ax, (name, vals) in zip(axes, panels):
            last_sc = draw_node_map(
                ax,
                data=data,
                values=vals,
                title=name,
                epsg_project=epsg_project,
                vmin=vmin,
                vmax=vmax,
            )

        for ax in axes[len(panels):]:
            ax.axis("off")

        fig.suptitle(f"{title}: {int(years[frame])}", fontsize=14)
        return []

    update(0)
    fig.colorbar(
        axes[0].collections[-1],
        ax=axes[:n_panels].tolist(),
        fraction=0.035,
        pad=0.02,
        label="Cumulative count",
    )

    anim = FuncAnimation(fig, update, frames=len(years), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("[SIM animation] saved:", out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--fit_json", default=None, type=str)
    args = parser.parse_args()

    cfg_named = load_config(args.config)

    fit_json = Path(args.fit_json) if args.fit_json else Path("out") / args.config / "fit_result.json"
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
        year_window=cfg_named["fit"].get("year_window", None),
    )

    params = SSSBFitParams(**fit_payload["params"])
    solver_cfg = SSSBFitConfig(**fit_payload["solver_config"])

    _, details = observed_driven_nll(
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

    sim_cfg = cfg_named.get("simulation", {})
    forecast_year = int(sim_cfg.get("forecast_year", int(data.years[-1]) + 10))
    seed = int(sim_cfg.get("seed", 2026))
    fps = int(sim_cfg.get("fps", 2))

    out_dir = Path("out") / args.config / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bass expected annual counts from observed aggregate curve, extended to forecast.
    t_obs, cum_obs = observed_monthly_curve(data.Y)
    bass = fit_bass_curve(t_obs, cum_obs)

    start_year = int(data.years[0])
    all_years = np.arange(start_year, forecast_year + 1)
    t_end = np.arange(1, all_years.size + 1, dtype=float)
    bass_cum_year_end = bass_cumulative(t_end, bass["p"], bass["q"], bass["M"])
    bass_cum_year_start = np.concatenate([[0.0], bass_cum_year_end[:-1]])
    annual_bass = np.maximum(bass_cum_year_end - bass_cum_year_start, 0.0)

    capacity = np.asarray(details["capacity"], dtype=float)
    
    print("[SIM diagnostics]")
    print("capacity sum:", float(np.sum(capacity)))
    print("capacity max:", float(np.max(capacity)))
    print("historical observed total:", float(data.Y.sum()))
    print("fitted expected historical total:", float(details["mu"].sum()))
    print("fit years:", int(data.years[0]), "to", int(data.years[-1]))
    print("forecast year:", forecast_year)
    print("S0:", params.S0)
    print("capacity sum:", capacity.sum())
    print("initial annual innovation mean:", (params.p * capacity).sum())
    
    sim_times = {}

    n_single_runs = int(sim_cfg.get("n_single_runs", 1))
    sssb_runs = []
    
    for rr in range(n_single_runs):
        run_seed = seed + rr
    
        t0 = time.perf_counter()
        sssb_rr = simulate_sssb_stochastic(
            data=data,
            params=params,
            solver_cfg=solver_cfg,
            capacity=capacity,
            years=data.years,
            forecast_year=forecast_year,
            seed=run_seed,
        )
        sim_times[f"sssb_run_{rr:02d}"] = time.perf_counter() - t0
        sssb_runs.append(sssb_rr)
    
    sssb = sssb_runs[0]

    n_nodes = data.Y.shape[1]
    uniform_weights = np.full(n_nodes, 1.0 / n_nodes)
    pop = np.clip(np.asarray(data.population, dtype=float), 0.0, None)
    pop_weights = pop / pop.sum() if pop.sum() > 0 else uniform_weights

    t0 = time.perf_counter()
    bass_uniform = simulate_bass_baseline(
        annual_expected=annual_bass,
        weights=uniform_weights,
        years=data.years,
        forecast_year=forecast_year,
        seed=seed + 1,
    )
    sim_times["bass_uniform"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    bass_population = simulate_bass_baseline(
        annual_expected=annual_bass,
        weights=pop_weights,
        years=data.years,
        forecast_year=forecast_year,
        seed=seed + 2,
    )
    sim_times["bass_population"] = time.perf_counter() - t0

    print("[SIM timing]")
    for name, val in sim_times.items():
        print(f"  {name}: {val:.4f} sec")
        
    run_batch = bool(sim_cfg.get("run_batch", True))

    if run_batch:
        n_batch = int(sim_cfg.get("batch_n_sims", 100))
    
        t0 = time.perf_counter()
        batch = simulate_sssb_stochastic_batch(
            data=data,
            params=params,
            solver_cfg=solver_cfg,
            capacity=capacity,
            years=data.years,
            forecast_year=forecast_year,
            seed=seed + 10_000,
            n_sims=n_batch,
        )
        batch_elapsed = time.perf_counter() - t0
        
        batch_uniform = simulate_bass_baseline_batch(
            annual_expected=annual_bass,
            weights=uniform_weights,
            years=data.years,
            forecast_year=forecast_year,
            seed=seed + 20_000,
            n_sims=n_batch,
        )
        
        batch_population = simulate_bass_baseline_batch(
            annual_expected=annual_bass,
            weights=pop_weights,
            years=data.years,
            forecast_year=forecast_year,
            seed=seed + 30_000,
            n_sims=n_batch,
        )
        
        sssb_mean = batch["mean_cum"]
        bass_uniform_mean = batch_uniform["mean_cum"]
        bass_population_mean = batch_population["mean_cum"]
    
        final_totals = np.asarray(batch["final_totals"], dtype=float)
    
        print("[SIM batch diagnostic]")
        print(f"  n_sims: {n_batch}")
        print(f"  total time: {batch_elapsed:.4f} sec")
        print(f"  time per simulation: {batch_elapsed / max(n_batch, 1):.6f} sec")
        print(f"  final total mean: {float(np.mean(final_totals)):.3f}")
        print(f"  final total std:  {float(np.std(final_totals)):.3f}")
        print(f"  final total min:  {float(np.min(final_totals)):.3f}")
        print(f"  final total max:  {float(np.max(final_totals)):.3f}")
        print(f"  final total nonzero fraction: {float(np.mean(final_totals > 0.0)):.3f}")
        print(f"  fitted expected historical total: {float(details['mu'].sum()):.3f}")
        print(f"  observed historical total: {float(data.Y.sum()):.3f}")

    observed_cum_hist = np.cumsum(data.Y, axis=0)

    fit_end_idx = len(data.years) - 1
    forecast_idx = len(all_years) - 1

    epsg_project = int(cfg_named["mesh"]["epsg_project"])

    # Plot 1
    save_static_panels(
        data=data,
        panels=[
            ("Observed cumulative", observed_cum_hist[fit_end_idx]),
            ("SSSB simulation", sssb["cum"][fit_end_idx]),
        ],
        out_path=out_dir / "fit_end_observed_vs_sssb.png",
        epsg_project=epsg_project,
        ncols=2,
        title=f"Cumulative adoptions through {int(data.years[-1])}",
    )

    # Plot 2
    save_static_panels(
        data=data,
        panels=[
            ("Observed cumulative", observed_cum_hist[fit_end_idx]),
            ("SSSB simulation", sssb["cum"][fit_end_idx]),
            ("Uniform Bass simulation", bass_uniform["cum"][fit_end_idx]),
            ("Population Bass simulation", bass_population["cum"][fit_end_idx]),
        ],
        out_path=out_dir / "fit_end_all_models.png",
        epsg_project=epsg_project,
        ncols=2,
        title=f"Cumulative adoptions through {int(data.years[-1])}",
    )

    # Plot 3
    save_static_panels(
        data=data,
        panels=[
            ("SSSB forecast simulation", sssb["cum"][forecast_idx]),
        ],
        out_path=out_dir / f"forecast_{forecast_year}_sssb.png",
        epsg_project=epsg_project,
        ncols=1,
        title=f"Forecast cumulative adoptions through {forecast_year}",
    )

    # Plot 4
    save_static_panels(
        data=data,
        panels=[
            ("SSSB forecast simulation", sssb["cum"][forecast_idx]),
            ("Uniform Bass forecast simulation", bass_uniform["cum"][forecast_idx]),
            ("Population Bass forecast simulation", bass_population["cum"][forecast_idx]),
        ],
        out_path=out_dir / f"forecast_{forecast_year}_all_models.png",
        epsg_project=epsg_project,
        ncols=3,
        title=f"Forecast cumulative adoptions through {forecast_year}",
    )
    
    # Batch Plots
    
    if run_batch:
        save_static_panels(
            data=data,
            panels=[
                ("Observed cumulative", observed_cum_hist[fit_end_idx]),
                ("SSSB simulation mean", sssb_mean[fit_end_idx]),
            ],
            out_path=out_dir / "fit_end_observed_vs_sssb_batch_mean.png",
            epsg_project=epsg_project,
            ncols=2,
            title=f"Cumulative adoptions through {int(data.years[-1])}: batch mean",
        )
    
        save_static_panels(
            data=data,
            panels=[
                ("Observed cumulative", observed_cum_hist[fit_end_idx]),
                ("SSSB simulation mean", sssb_mean[fit_end_idx]),
                ("Uniform Bass simulation mean", bass_uniform_mean[fit_end_idx]),
                ("Population Bass simulation mean", bass_population_mean[fit_end_idx]),
            ],
            out_path=out_dir / "fit_end_all_models_batch_mean.png",
            epsg_project=epsg_project,
            ncols=2,
            title=f"Cumulative adoptions through {int(data.years[-1])}: batch mean",
        )
    
        save_static_panels(
            data=data,
            panels=[
                ("SSSB forecast simulation mean", sssb_mean[forecast_idx]),
            ],
            out_path=out_dir / f"forecast_{forecast_year}_sssb_batch_mean.png",
            epsg_project=epsg_project,
            ncols=1,
            title=f"Forecast cumulative adoptions through {forecast_year}: batch mean",
        )
    
        save_static_panels(
            data=data,
            panels=[
                ("SSSB forecast simulation mean", sssb_mean[forecast_idx]),
                ("Uniform Bass forecast simulation mean", bass_uniform_mean[forecast_idx]),
                ("Population Bass forecast simulation mean", bass_population_mean[forecast_idx]),
            ],
            out_path=out_dir / f"forecast_{forecast_year}_all_models_batch_mean.png",
            epsg_project=epsg_project,
            ncols=3,
            title=f"Forecast cumulative adoptions through {forecast_year}: batch mean",
        )

    # Animations
    hist_years = data.years
    forecast_years = all_years

    save_animation(
        data=data,
        panels_by_year=[
            [
                ("Observed cumulative", observed_cum_hist[k]),
                ("SSSB simulation", sssb["cum"][k]),
            ]
            for k in range(len(hist_years))
        ],
        years=hist_years,
        out_path=out_dir / "anim_fit_observed_vs_sssb.gif",
        epsg_project=epsg_project,
        ncols=2,
        title="Historical cumulative adoptions",
        fps=fps,
    )

    save_animation(
        data=data,
        panels_by_year=[
            [
                ("Observed cumulative", observed_cum_hist[k]),
                ("SSSB simulation", sssb["cum"][k]),
                ("Uniform Bass simulation", bass_uniform["cum"][k]),
                ("Population Bass simulation", bass_population["cum"][k]),
            ]
            for k in range(len(hist_years))
        ],
        years=hist_years,
        out_path=out_dir / "anim_fit_all_models.gif",
        epsg_project=epsg_project,
        ncols=2,
        title="Historical cumulative adoptions",
        fps=fps,
    )

    save_animation(
        data=data,
        panels_by_year=[
            [("SSSB forecast simulation", sssb["cum"][k])]
            for k in range(len(forecast_years))
        ],
        years=forecast_years,
        out_path=out_dir / f"anim_forecast_{forecast_year}_sssb.gif",
        epsg_project=epsg_project,
        ncols=1,
        title="Forecast cumulative adoptions",
        fps=fps,
    )

    save_animation(
        data=data,
        panels_by_year=[
            [
                ("SSSB forecast simulation", sssb["cum"][k]),
                ("Uniform Bass forecast simulation", bass_uniform["cum"][k]),
                ("Population Bass forecast simulation", bass_population["cum"][k]),
            ]
            for k in range(len(forecast_years))
        ],
        years=forecast_years,
        out_path=out_dir / f"anim_forecast_{forecast_year}_all_models.gif",
        epsg_project=epsg_project,
        ncols=3,
        title="Forecast cumulative adoptions",
        fps=fps,
    )

    print("[SIM] wrote outputs to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())