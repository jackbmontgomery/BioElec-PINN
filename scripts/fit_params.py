from jax import vmap
import jax.numpy as jnp
import optimistix as optx
import equinox as eqx
from bioelec_pinn import Parameters, solve, ODEModel, FixedParameters, State
from bioelec_pinn.cycle_data import (
    CYCLE_S_IN,
    CYCLE_DURATION,
    CYCLE_ALPHA_0,
    CYCLE_PARAMETERS,
)
import jax
import optax
import polars as pl
import matplotlib.pyplot as plt

voltage_data = pl.read_csv("./data/pardogomez2026_fig4_voltage_experimental.csv")
cod = pl.read_csv("./data/pardogomez2026_fig5_cod_experimental.csv")


def make_loss_fn(model: ODEModel, y0, t1, ts, voltage):
    def loss_fn(params: Parameters):
        sol = solve(model, params, y0, t1, saveat_ts=ts)
        I_pred = vmap(model.current, in_axes=(0, None))(sol.ys, params)
        E_pred = model.fp.R_ext * I_pred
        return jnp.mean(jnp.square(E_pred - voltage))

    return loss_fn


cycle = 1

cycle_voltage_data = voltage_data.filter(pl.col("cycle") == cycle)
ts = cycle_voltage_data["time_d"].to_jax()
v = cycle_voltage_data["voltage_V"].to_jax()

S_in = CYCLE_S_IN[cycle]
t1 = CYCLE_DURATION[cycle]
alpha = CYCLE_ALPHA_0[cycle]
params = CYCLE_PARAMETERS[cycle]


fp = FixedParameters(S_in=S_in)
root_finder = optx.Newton(rtol=1e-8, atol=1e-6)
y0 = State(
    S_bulk=jnp.asarray(S_in),
    S_bio=jnp.asarray(S_in * alpha),
    x_p=jnp.asarray(0.7),
    x_s=jnp.asarray(0.8),
    M_ox=jnp.asarray(0.5),
)
model = ODEModel(fp, root_finder)

opt = optax.adam(1e-3)
opt_state = opt.init(eqx.filter(params, eqx.is_array))
loss_fn = eqx.filter_value_and_grad(make_loss_fn(model, y0, t1, ts, v))


@eqx.filter_jit
def step(params, opt_state):
    loss_val, grads = loss_fn(params)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(params, eqx.is_array))
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val


for i in range(20):
    params, opt_state, loss_val = step(params, opt_state)
    print(i, loss_val)

sol = solve(model, params, y0, t1, ts)
I_pred = vmap(model.current, in_axes=(0, None))(sol.ys, params)
E_pred = model.fp.R_ext * I_pred
plt.plot(sol.ts, E_pred)
plt.scatter(ts, v)
plt.savefig("figures/cycle0.png")
