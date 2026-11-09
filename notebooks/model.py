import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from bioelec_pinn import CYCLE_DURATION, cycle_setup, solve

voltage = pl.read_csv("./data/pardogomez2026_fig4_voltage_experimental.csv")
cod = pl.read_csv("./data/pardogomez2026_fig5_cod_experimental.csv")


def run(cycle, n=400, **model_kwargs):
    model, params, y0 = cycle_setup(cycle, **model_kwargs)
    ts = jnp.linspace(0.0, CYCLE_DURATION[cycle], n)
    sol = solve(model, params, y0, CYCLE_DURATION[cycle], saveat_ts=ts)
    V = np.asarray(jax.vmap(lambda y: model.voltage(y, params))(sol.ys))
    return np.asarray(ts), sol.ys, V


trajectories = {c: run(c) for c in (1, 2, 3, 4)}


fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex="col")
for i, c in enumerate((1, 2, 3, 4)):
    ts, ys_1, V = trajectories[c]

    ax = axes[0, i]
    v = voltage.filter(pl.col("cycle") == c)
    ax.plot(v["time_d"], v["voltage_V"], ".", ms=2, label="experimental")
    ax.plot(ts, V, label="forward solve")
    ax.set_title(f"Cycle {c}")
    ax.set_ylabel("voltage [V]")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8)

    ax = axes[1, i]
    d = cod.filter(pl.col("cycle") == c)
    ax.plot(d["time_d"], d["cod_mg_per_L"], "o", label="experimental COD")
    ax.plot(ts, ys_1.S_bulk * 1000, label=r"$S_{bulk}$")
    ax.plot(ts, ys_1.S_bio * 1000, "--", label=r"$S_{bio}$")
    ax.set_xlabel("time [d]")
    ax.set_ylabel("COD [mg/L]")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig("./1.png")


fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
for c in (1, 2, 3, 4):
    ts, ys, _ = trajectories[c]
    axes2[0].plot(ts, ys.M_ox, label=f"cycle {c}")
    axes2[1].plot(ts, ys.x_p, label=f"cycle {c}")
    axes2[2].plot(ts, ys.x_s, label=f"cycle {c}")
for ax, name in zip(axes2, (r"$M_{ox}$ [-]", r"$x_p$ [g/L]", r"$x_s$ [g/L]")):
    ax.set_xlabel("time [d]")
    ax.set_ylabel(name)
    ax.legend(fontsize=8)
axes2[0].set_yscale("log")
fig2.savefig("./2.png")
