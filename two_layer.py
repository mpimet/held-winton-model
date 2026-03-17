# %%
import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr


sec_per_year = 365.25 * 24 * 3600


# 3. Forcings
def F_aerosol(t, t1y=10, t2y=40, d1y=3, d2y=2, A=1.2):
    t1 = sec_per_year * t1y
    t2 = sec_per_year * t2y
    d1 = sec_per_year * d1y
    d2 = sec_per_year * d2y
    return -A * (
        (np.atan2(t - t1, d1) / np.pi - np.atan2(t - t2, d2) / np.pi * 1 / 2.0) + 0.25
    )


def F_CO2(C):
    return 5.35 * np.log(C / 280.0)


# 4. Held Winton Model
def held_winton(t, y, C, C0, lam_lw, lam_sw, kappa, lam0, dFdt, F_abrupt, A):
    T, T_0 = y
    lam = lam_lw + lam_sw
    F = F_abrupt + dFdt * t + F_aerosol(t, A=A)
    dTdt = (F + lam * T - kappa * (T - T_0) - lam0 * (T - T_0)) / C
    dT0dt = (kappa * (T - T_0)) / C0
    return [dTdt, dT0dt]


def run_two_layer_model(
    t_span, C, C0, lam_lw, lam_sw, kappa, lam0, dFdt, F_0, A, y0=[0, 0]
):
    t_eval = np.linspace(t_span[0], t_span[1], 50000)
    sol = solve_ivp(
        held_winton,
        t_span,
        y0,
        args=(C, C0, lam_lw, lam_sw, kappa, lam0, dFdt, F_0, A),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    F_lw = sol.t * dFdt + F_0
    F_sw = F_aerosol(sol.t, A=A)
    F = F_lw + F_sw
    N_lw = F_lw + sol.y[0] * lam_lw - lam0 * (sol.y[0] - sol.y[1])
    N_sw = F_sw + sol.y[0] * lam_sw
    N = N_sw + N_lw
    return xr.Dataset(
        dict(
            T_sfc=(
                "time",
                sol.y[0],
                {"long_name": "surface temperature", "units": "K"},
            ),
            T_deep=(
                "time",
                sol.y[1],
                {"long_name": "deep ocean temperature", "units": "K"},
            ),
            F_lw=("time", F_lw, {"long_name": "longwave forcing", "units": "W m-2"}),
            F_sw=("time", F_sw, {"long_name": "shortwave forcing", "units": "W m-2"}),
            F=("time", F, {"long_name": "net forcing", "units": "W m-2"}),
            N_lw=("time", N_lw, {"long_name": "longwave imbalance", "units": "W m-2"}),
            N_sw=("time", N_sw, {"long_name": "shortwave imbalance", "units": "W m-2"}),
            N=("time", N, {"long_name": "net imbalance", "units": "W m-2"}),
        ),
        coords=dict(
            time=("time", sol.t / sec_per_year, {"long_name": "time", "units": "year"}),
        ),
    )
