from __future__ import annotations


DEFAULT = {
    "paths": {
        "admin1_shp": "data/raw/ne_10m_admin_1_states_provinces_lakes/ne_10m_admin_1_states_provinces_lakes.shp",
        "county_shp": "data/raw/cb_2023_us_county_5m/cb_2023_us_county_5m.shp",
        "pop_csv": "data/raw/zip_population_all.csv",
        "lspv_csv": "data/raw/uspvdb_v4_0_20260414.csv",
        "transmission_shp": "data/raw/Transmission_Lines/Transmission_Lines.shp",
        "pvout_tif": "data/raw/PVOUT.tif",
    },

    "region": {
        "states": [],
        "counties": [],
    },

    "mesh": {
        "h_km": 25.0,
        "simplify_km": 5.0,
        "epsg_project": 5070,
        "overwrite_mesh": False,
    },

    "density": {
        "smooth_length_km": 75.0,
        "smooth_k_neighbors": 32,
        "smooth_kernel": "gaussian",
        "transmission_buffer_km": 15,
        "plot_pop_year": 2020,
        "adoption_plot_scale": "log1p",  # "linear", "log1p" or "mixed"
        "mixed_log_range": 8.0,
        "adoption_plot_quantile": 1.0,  # use 1.0 for true max; 0.99 for clipped scale
        "adoption_shared_colorbar": True,
    },

    "fit": {
        "dt_years": 0.05,
        "use_covariates": True,
        "fit_S0": False,
        "n_random": 500,
        "maxiter": 500,
        "seed": 1337,
        "population_key": "population_smooth_2020",
        "progress_freq": 10,
        "year_window": None,  # e.g. [2007, 2025], inclusive
    },

    "param_bounds": {
        "p": [1e-12, 1e-3],
        "q": [1e-5, 10],
        "gamma_J": [100, 1e5],
        "k_J": [1e-3, 1e3],
        "D": [1e-3, 1e3],
        "S0": [0, 0],
        "r0": [1e-15, 1e-10],
        "r1": [1e-3, 1e+3],
        "r2": [1e-15, 1e-10],
        "FI_a": [0.1, 10],
        "FI_b": [0.1, 10],
        "FI_c": [0.01, 10],
    },

    "initial": {
        "p": 1e-2,
        "q": 1e-1,
        "gamma_J": 1e-1,
        "k_J": 0.01,
        "D": 100,
        "S0": 0,
        "r0": 1e-5,
        "r1": 1e-9,
        "r2": 1e-6,
        "FI_a": 1,
        "FI_b": 1,
        "FI_c": 1,
    },
}


CONFIGS = {
    "default": {},
    
    "CA_NV_AZ_UT_config": {
        "region": {
            "states": ["CA", "NV", "AZ", "UT"],
            "counties": [],
        },
        "mesh": {
            "h_km": 6,
            "simplify_km": 18,
        },
        "density": {
            "smooth_length_km": 75,
            "smooth_kernel": "gaussian",
            "transmission_buffer_km": 100,
        },
        "fit": {
            "year_window": [2007, 2025],
            "use_covariates": True,
            "fit_S0": False,
            "n_random": 2000, #10000
            "maxiter": 500,
            "dt_years": 0.1,
        },
    },
}