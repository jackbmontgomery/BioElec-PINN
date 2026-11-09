"""Mechanistic DAE model of Pardo Gomez et al. (2026), Eqs. (1)-(15).

Unit convention follows the paper's Tables 4-5 throughout:
    substrate, biomass  [g/L]
    rates               [1/d]
    gamma               [g/mol]
    K_r1, K_r2          [L/g]
    I_mfc               [A]
Note that gamma and x_p must use the *same* mass unit for Eq. (5) to be
dimensionally consistent, so COD data in mg/L must be divided by 1000.
"""

import diffrax
import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike

F = 96_485.0  # [C/mol]
R = 8.314  # [J/mol/K]
SECONDS_PER_DAY = 86_400.0


class State(eqx.Module):
    S_bulk: Array
    S_bio: Array
    x_p: Array
    x_s: Array
    M_ox: Array


class FixedParameters(eqx.Module):
    """Table 4: fixed parameters and physical constants."""

    S_in: float
    T: float = 298.15  # [K] Liu et al. (2021)
    V: float = 0.1  # [L] Liu et al. (2021)
    A_n: float = 4.56e-3  # [m^2] Liu et al. (2021)
    R_ext: float = 10_000.0  # [Ohm] Liu et al. (2021)
    beta: float = 0.5  # [-] Gadkari et al. (2019)
    i_0: float = 0.048  # [A/m^2] Gadkari et al. (2019)
    m: float = 2.0  # [-] Pinto et al. (2010)
    gamma: float = 663.4  # [g/mol] Pinto et al. (2010)
    E_min: float = 0.01  # [V] Gadkari et al. (2019)
    # NOTE: E_max is NOT given in the paper's Table 4. 0.66 V is Gadkari et
    # al. (2019)'s value, but the measured Cycle 3 plateau reaches 0.677 V,
    # which this bound cannot reproduce. Treat it as uncertain.
    E_max: float = 0.66  # [V] assumed, see note
    R_min: float = 8.0  # [Ohm] Gadkari et al. (2019)
    R_max: float = 500.0  # [Ohm] Gadkari et al. (2019)
    K_Sp: float = 0.103  # [g/L] Gadkari et al. (2019)
    decay_factor: float = 0.02  # [-] Pinto et al. (2010)

    def K_dp(self, p: "Parameters") -> Array:
        # Table 4: K_d,p = 0.02 * mu_max,p
        return self.decay_factor * p.mu_max_p

    def K_ds(self, p: "Parameters") -> Array:
        # Table 4: K_d,s = 0.02 * mu_max,s
        return self.decay_factor * p.mu_max_s


class Parameters(eqx.Module):
    """Table 5: the ten cycle-specific parameters estimated by the PINN."""

    q_max_p: Array
    mu_max_p: Array
    q_max_s: Array
    mu_max_s: Array
    K_Ss: Array
    K_M: Array
    Y_med: Array
    k_mt: Array
    K_r1: Array
    K_r2: Array

    def __init__(
        self,
        q_max_p: ArrayLike,
        mu_max_p: ArrayLike,
        q_max_s: ArrayLike,
        mu_max_s: ArrayLike,
        K_Ss: ArrayLike,
        K_M: ArrayLike,
        Y_med: ArrayLike,
        k_mt: ArrayLike,
        K_r1: ArrayLike,
        K_r2: ArrayLike,
    ):
        self.q_max_p = jnp.asarray(q_max_p)
        self.mu_max_p = jnp.asarray(mu_max_p)
        self.q_max_s = jnp.asarray(q_max_s)
        self.mu_max_s = jnp.asarray(mu_max_s)
        self.K_Ss = jnp.asarray(K_Ss)
        self.K_M = jnp.asarray(K_M)
        self.Y_med = jnp.asarray(Y_med)
        self.k_mt = jnp.asarray(k_mt)
        self.K_r1 = jnp.asarray(K_r1)
        self.K_r2 = jnp.asarray(K_r2)


class Model(eqx.Module):
    fp: FixedParameters

    # The paper writes a bare "S" in Eqs. (12), (14), (15) without saying
    # whether it is S_bulk or S_bio. Default to the biofilm concentration,
    # since the electrochemistry is local to the anode.
    electro_uses_bio: bool = eqx.field(static=True, default=True)

    def _S_electro(self, y: State) -> Array:
        return y.S_bio if self.electro_uses_bio else y.S_bulk

    def kinetic_rates(
        self, y: State, p: Parameters
    ) -> tuple[Array, Array, Array, Array]:
        # Eqs. (7)-(8): primary population, dual Monod on biofilm substrate
        # and oxidized mediator.
        rate_p = (y.S_bio / (self.fp.K_Sp + y.S_bio)) * (y.M_ox / (p.K_M + y.M_ox))
        q_p = p.q_max_p * rate_p
        mu_p = p.mu_max_p * rate_p

        # Eqs. (9)-(10): secondary population, single Monod on bulk substrate.
        rate_s = y.S_bulk / (p.K_Ss + y.S_bulk)
        q_s = p.q_max_s * rate_s
        mu_s = p.mu_max_s * rate_s

        return q_p, mu_p, q_s, mu_s

    def ocv(self, y: State, p: Parameters) -> Array:
        # Eq. (14)
        S = self._S_electro(y)
        return self.fp.E_min + (self.fp.E_max - self.fp.E_min) * jnp.exp(
            -1.0 / (p.K_r1 * S)
        )

    def internal_resistance(self, y: State, p: Parameters) -> Array:
        # Eq. (15)
        S = self._S_electro(y)
        return self.fp.R_min + (self.fp.R_max - self.fp.R_min) * jnp.exp(-p.K_r2 * S)

    def eta_conc(self, y: State) -> Array:
        # Eq. (12)
        S = self._S_electro(y)
        return R * self.fp.T * jnp.log(self.fp.S_in / S) / (self.fp.m * F)

    def eta_act(self, I_mfc: Array) -> Array:
        # Eq. (13)
        prefactor = R * self.fp.T / (self.fp.beta * self.fp.m * F)
        return prefactor * jnp.arcsinh(I_mfc / (self.fp.A_n * self.fp.i_0))

    def current(self, y: State, p: Parameters) -> Array:
        """Solve the algebraic constraint of Eq. (11) for I_MFC."""
        E_ocv = self.ocv(y, p)
        R_int = self.internal_resistance(y, p)
        eta_conc = self.eta_conc(y)
        R_tot = self.fp.R_ext + R_int

        def residual(I_mfc, _):
            # Eq. (33), the well-scaled residual form, normalised by R_tot so
            # the root-find operates on a quantity of order I_mfc.
            return ((E_ocv - self.eta_act(I_mfc) - eta_conc) - I_mfc * R_tot) / R_tot

        # Ohmic solution ignoring activation loss: an excellent initial guess
        # because eta_act << E_ocv for R_ext = 10 kOhm.
        I_guess = (E_ocv - eta_conc) / R_tot
        solver = optx.Newton(rtol=1e-10, atol=1e-14)
        sol = optx.root_find(residual, solver, y0=I_guess, max_steps=64)
        return sol.value

    def voltage(self, y: State, p: Parameters) -> Array:
        # Eq. (35)
        return self.current(y, p) * self.fp.R_ext

    def __call__(self, t: float, y: State, p: Parameters) -> State:
        q_p, mu_p, q_s, mu_s = self.kinetic_rates(y, p)
        I_mfc = self.current(y, p)

        # Eqs. (1)-(2)
        flux = p.k_mt * (y.S_bulk - y.S_bio)
        dS_bulk = -flux - q_s * y.x_s
        dS_bio = flux - q_p * y.x_p

        # Eqs. (3)-(4)
        dx_p = (mu_p - self.fp.K_dp(p)) * y.x_p
        dx_s = (mu_s - self.fp.K_ds(p)) * y.x_s

        # Eq. (5). The current term is per second, so convert it to per day.
        dM_ox = -p.Y_med * q_p + (
            self.fp.gamma
            * I_mfc
            * SECONDS_PER_DAY
            / (self.fp.V * y.x_p * self.fp.m * F)
        )

        return State(S_bulk=dS_bulk, S_bio=dS_bio, x_p=dx_p, x_s=dx_s, M_ox=dM_ox)


def solve(
    model: Model,
    p: Parameters,
    y0: State,
    t1: float,
    saveat_ts: Array | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 1_000_000,
):
    saveat = (
        diffrax.SaveAt(ts=saveat_ts)
        if saveat_ts is not None
        else diffrax.SaveAt(t1=True)
    )
    return diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        diffrax.Kvaerno5(root_finder=optx.Newton(rtol=rtol, atol=atol)),
        t0=0.0,
        t1=t1,
        dt0=1e-4,
        y0=y0,
        args=p,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )


# Table 5 (Cycle 1) + Table 4 initial conditions.
CYCLE_PARAMETERS = {
    1: Parameters(
        q_max_p=0.292,
        mu_max_p=0.12,
        q_max_s=0.318,
        mu_max_s=1.29,
        K_Ss=0.255,
        K_M=0.0102,
        Y_med=49.2,
        k_mt=1.01,
        K_r1=600.4,
        K_r2=12.21,
    ),
    2: Parameters(
        q_max_p=0.618,
        mu_max_p=0.198,
        q_max_s=0.321,
        mu_max_s=1.006,
        K_Ss=0.945,
        K_M=0.82,
        Y_med=41.86,
        k_mt=3.147,
        K_r1=314.5,
        K_r2=38.85,
    ),
    3: Parameters(
        q_max_p=4.341,
        mu_max_p=2.163,
        q_max_s=4.125,
        mu_max_s=0.865,
        K_Ss=5.69,
        K_M=0.395,
        Y_med=18.56,
        k_mt=4.263,
        K_r1=1714.3,
        K_r2=7.459,
    ),
    4: Parameters(
        q_max_p=2.856,
        mu_max_p=3.366,
        q_max_s=1.861,
        mu_max_s=0.313,
        K_Ss=5.112,
        K_M=0.414,
        Y_med=16.07,
        k_mt=5.456,
        K_r1=405.4,
        K_r2=15.0,
    ),
}

# Table 3: *measured* initial COD, not the nominal 1/2/3/4 g/L labels.
CYCLE_S_IN = {1: 1.000, 2: 1.651, 3: 2.436, 4: 3.619}  # [g/L]
CYCLE_S_FINAL = {1: 0.223, 2: 0.266, 3: 0.338, 4: 0.414}  # [g/L], for reference
CYCLE_ALPHA_0 = {1: 0.3, 2: 0.70, 3: 0.99, 4: 0.995}  # Table 4
CYCLE_DURATION = {  # Table 3: 96, 131, 240, 383 h
    1: 96.0 / 24.0,
    2: 131.0 / 24.0,
    3: 240.0 / 24.0,
    4: 383.0 / 24.0,
}


def cycle_setup(cycle: int, **model_kwargs) -> tuple[Model, Parameters, State]:
    S_in = CYCLE_S_IN[cycle]
    fp = FixedParameters(S_in=S_in)
    y0 = State(
        S_bulk=jnp.asarray(S_in),
        S_bio=jnp.asarray(S_in * CYCLE_ALPHA_0[cycle]),
        x_p=jnp.asarray(0.7),
        x_s=jnp.asarray(0.8),
        M_ox=jnp.asarray(0.5),
    )
    return Model(fp=fp, **model_kwargs), CYCLE_PARAMETERS[cycle], y0
