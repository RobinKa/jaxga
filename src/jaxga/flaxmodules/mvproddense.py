from ..ops.multiply import get_mv_multiply
from ..signatures import positive_signature
from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp


class MVProdDense(nn.Module):
    x_ind: Sequence[Sequence[str]]
    w_ind: Sequence[Sequence[str]]
    units: int
    w_init: Callable = nn.initializers.lecun_normal()
    signature: Callable[[int], float] = positive_signature
    prod: str = "gp"

    def setup(self):
        self.mv_multiply, self.result_indices = get_mv_multiply(
            self.x_ind, self.w_ind, signature=self.signature, prod=self.prod
        )

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "w", self.w_init,
            (len(self.w_ind), x.shape[-1], self.units)
        )
        x = jnp.expand_dims(x, -1)
        # [M, ...B, I] * [M, I, O] -> [M, ...B, I, O] -> [M, ...B, O]
        result_values_unreduced = self.mv_multiply(x, w)
        result = jnp.sum(result_values_unreduced, axis=-2)
        return result
