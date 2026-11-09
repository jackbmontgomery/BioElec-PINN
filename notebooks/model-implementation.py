"""Forward integration of the Pardo Gomez et al. (2026) mechanistic model.

Reproduces the paper's "classical solver" check: integrate Eqs. (1)-(5) with
the algebraic current constraint Eq. (11), using the Table 5 cycle-specific
parameters, and compare against the digitised Fig. 4 / Fig. 5 observations.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import polars as pl

from bioelec_pinn import CYCLE_DURATION, cycle_setup, solve

DATA = "data"
# Table 6, classical-solver column, for reference.
PAPER_NRMSE = {
    1: (0.111, 0.088),
    2: (0.117, 0.183),
    3: (0.088, 0.192),
    4: (0.067, 0.226),
}


def nrmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)) / (obs.max() - obs.min()))


voltage = pl.read_csv(f"{DATA}/pardogomez2026_fig4_voltage_experimental.csv")
cod = pl.read_csv(f"{DATA}/pardogomez2026_fig5_cod_experimental.csv")

for cycle in (1, 2, 3, 4):
    model, params, y0 = cycle_setup(cycle)

    v = voltage.filter(pl.col("cycle") == cycle)
    c = cod.filter(pl.col("cycle") == cycle)
    t_v = jnp.asarray(v["time_d"].to_numpy())
    t_c = jnp.asarray(c["time_d"].to_numpy())
    ts = jnp.sort(jnp.concatenate([t_v, t_c]))
    t1 = max(CYCLE_DURATION[cycle], float(ts[-1])) + 1e-9

    sol = solve(model, params, y0, t1, saveat_ts=ts)
    V_pred = np.asarray(jax.vmap(lambda y: model.voltage(y, params))(sol.ys))
    S_pred = np.asarray(sol.ys.S_bulk) * 1000.0  # g/L -> mg/L

    tsn = np.asarray(ts)
    iv = np.searchsorted(tsn, np.asarray(t_v))
    ic = np.searchsorted(tsn, np.asarray(t_c))

    nv = nrmse(V_pred[iv], v["voltage_V"].to_numpy())
    nc = nrmse(S_pred[ic], c["cod_mg_per_L"].to_numpy())
    pv, pc = PAPER_NRMSE[cycle]
    print(
        f"cycle {cycle}: V NRMSE={nv:.3f} (paper {pv:.3f})   "
        f"COD NRMSE={nc:.3f} (paper {pc:.3f})   "
        f"V range {V_pred.min():.3f}-{V_pred.max():.3f} V"
    )
