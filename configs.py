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
        "adoption_plot_scale": "linear",  # "linear" or "log1p"
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
    },

    "param_bounds": {
        "p": [1e-18, 1e-1],
        "q": [1e-13, 1],
        "gamma_J": [1e-8, 100],
        "k_J": [1e-6, 10],
        "D": [1e-5, 1000],
        "S0": [0, 0],
        "r0": [1e-18, 1],
        "r1": [1e-18, 1],
        "r2": [1e-18, 1],
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
    },
}


CONFIGS = {
    "default": {},

    "CA_NV_AZ_config": {
        "region": {
            "states": ["CA", "NV", "AZ"],
            "counties": [],
        },
        "mesh": {
            "h_km": 8,
            "simplify_km": 24,
        },
        "density": {
            "smooth_length_km": 75,
            "smooth_kernel": "gaussian",
            "transmission_buffer_km": 25,
        },
        "fit": {
            "use_covariates": True,
            "fit_S0": False,
            "n_random": 500,
            "maxiter": 500,
            "dt_years": 0.05,
        },
    },
    
    "CA_NV_AZ_UT_config": {
        "region": {
            "states": ["CA", "NV", "AZ", "UT"],
            "counties": [],
        },
        "mesh": {
            "h_km": 8,
            "simplify_km": 24,
        },
        "density": {
            "smooth_length_km": 75,
            "smooth_kernel": "gaussian",
            "transmission_buffer_km": 25,
        },
        "fit": {
            "use_covariates": True,
            "fit_S0": False,
            "n_random": 5000, #10000
            "maxiter": 500,
            "dt_years": 0.05,
        },
    },
}