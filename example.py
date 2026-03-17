# %%
import numpy as np
from scipy.integrate import solve_ivp
import moist_thermodynamics.constants as mtc
import matplotlib.pyplot as plt
import seaborn as sns
import two_layer
from two_layer import sec_per_year

# 1. Physical constants
liq_specific_heat = mtc.liquid_water_specific_heat
liq_density = 1025

# 3. Model Parameters
h = 40  # Mixed layer depth (m)
H = 3600  # Mean ocean equivalent depth
C = liq_specific_heat * liq_density * h
C0 = C * (H / h - 1)


# 1. Initial Conditions & Forcing
y0 = [0.0, 0.0]  # Initial temperature anomalies (K)

F2x = two_layer.F_CO2(2 * 280)  # 2xCO2 forcing
t2x = 70 * sec_per_year
dFdt = 0.0 * F2x / t2x  # Radiative forcing (e.g., 2xCO2 step change)
F_0 = F2x
A = 0.0  # aerosol forcing

# 2. Parameters
lam_sw = 1.0
lam_lw = -2.5
lam = lam_lw + lam_sw  # Climate feedback parameter (W m^-2 K^-1)
lam0 = -0.0  # Pattern Effect
kappa = 0.5  # Deep ocean heat uptake coefficient (W m^-2 K^-1)

# 2. Solve the IVP
years = 7000
end_time = years * sec_per_year
t = np.linspace(0, end_time, 1000)
t_span = (0, end_time)

ds = two_layer.run_two_layer_model(
    t_span=t_span,
    C=C,
    C0=C0,
    lam_lw=lam_lw,
    lam_sw=lam_sw,
    kappa=kappa,
    lam0=lam0,
    dFdt=dFdt,
    F_0=F_0,
    A=A,
)

# %%

# 5. Plot the Evolution
sns.set_context("talk")
plt.figure(figsize=(7, 4))


ds.T_sfc.plot(c="tab:red", label="Surface", lw=2)
ds.T_deep.plot(c="tab:blue", label="Deep Ocean", lw=2)
plt.axhline(y=-F2x / lam, color="gray", linestyle="--", label="$-F/\\lambda$")

plt.title("Held-Winton Two-Layer Model Response")
plt.xlabel("t / yr")
plt.ylabel("$T$ / K")
plt.legend()
sns.despine(offset=10)
plt.show()
# %%

sns.set_context("talk")

plt.figure(figsize=(10, 6))
ds.F.plot(color="tab:red", label="F", lw=2)
ds.N.plot(color="k", label="N", lw=2)
ds.N_lw.plot(label="$N_\\mathrm{LW}$", color="k", lw=2, ls="--")
ds.N_sw.plot(label="$N_\\mathrm{SW}$", color="k", lw=2, ls=":")
plt.plot(
    [10, 50],
    [0, -0.45 * 4],
    label="$N_\\mathrm{LW,obs}$",
    color="grey",
    lw=1,
    ls="solid",
)

plt.title("Energy Imbalance")
plt.xlabel("t / yr")
plt.ylabel("$N$ / Wm$^{-2}$")
plt.legend(ncol=2)
sns.despine(offset=10)
plt.show()
# %%

sns.set_context("talk")


plt.figure(figsize=(6, 4))

plt.plot(ds.T_sfc, ds.N, label="N", color="k", lw=2)

plt.title("Gregory Plot")
plt.xlabel("$T$/ K")
plt.ylabel("$N$ / Wm$^{-2}$")
plt.gca().set_ylim(0, None)
sns.despine(offset=10)
plt.show()

# %%
sns.set_context("talk")
plt.figure(figsize=(5, 3))

plt.title("Aerosol Forcing")
plt.plot(t / sec_per_year, two_layer.F_aerosol(t, A=A))
plt.xlabel("t / yr")
plt.ylabel("$F_\\mathrm{aer}$ / Wm$^{-2}$")
sns.despine(offset=10)
plt.show()
