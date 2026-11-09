from bioelec_pinn.model import (
    FixedParameters,
    ODEModel,
    Parameters,
    State,
    solve,
)
import jax

jax.config.update("jax_enable_x64", True)

__all__ = [
    "FixedParameters",
    "ODEModel",
    "Parameters",
    "State",
    "solve",
]
