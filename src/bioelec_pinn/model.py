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

    # NOTE: E_max is NOT given in the paper's Table 4 or 5. 0.66 V is Gadkari et
    # al. (2019)'s value. So, we default to this but the experimental data has max
    # voltage that is more than this. We aim to also estimte this parameter.

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
    E_max: Array

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
        E_max: ArrayLike = 0.66,  # [V] Gadkari et al. (2019)
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
        self.E_max = jnp.asarray(E_max)


class ODEModel(eqx.Module):
    fp: FixedParameters
    root_finder: optx.AbstractRootFinder

    # NOTE: The paper writes just "S" in Eqs. (12), (14), (15) without
    # specifying if its bulk of bio substrate concentration. We
    # choose biofilm since the electrochemistry is local to the anode.

    def kinetic_rates(
        self, y: State, p: Parameters
    ) -> tuple[Array, Array, Array, Array]:
        # Eqs. (7)-(8): primary population, assumed S_bio for S
        rate_p = (y.S_bio / (self.fp.K_Sp + y.S_bio)) * (y.M_ox / (p.K_M + y.M_ox))
        q_p = p.q_max_p * rate_p
        mu_p = p.mu_max_p * rate_p

        # Eqs. (9)-(10): secondary population, assumed S_bulk for S
        rate_s = y.S_bulk / (p.K_Ss + y.S_bulk)
        q_s = p.q_max_s * rate_s
        mu_s = p.mu_max_s * rate_s

        return q_p, mu_p, q_s, mu_s

    def ocv(self, y: State, p: Parameters) -> Array:
        # Eq. (14)
        return self.fp.E_min + (p.E_max - self.fp.E_min) * jnp.exp(
            -1.0 / (p.K_r1 * y.S_bulk)
        )

    def internal_resistance(self, y: State, p: Parameters) -> Array:
        # Eq. (15)
        return self.fp.R_min + (self.fp.R_max - self.fp.R_min) * jnp.exp(
            1 / (p.K_r2 * y.S_bulk)
        )

    def eta_conc(self, y: State) -> Array:
        # Eq. (12)
        return R * self.fp.T * jnp.log(self.fp.S_in / y.S_bulk) / (self.fp.m * F)

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
            # Eq. (33)
            return ((E_ocv - self.eta_act(I_mfc) - eta_conc) - I_mfc * R_tot) / R_tot

        I_guess = (E_ocv - eta_conc) / R_tot
        sol = optx.root_find(residual, self.root_finder, y0=I_guess, max_steps=64)
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
    model: ODEModel,
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
