from .ops.inverse import get_mv_inverse
from .ops.dual import get_mv_dual
from .ops.reduce_same import get_mv_reduce_same
from .ops.keepnonzero import get_mv_keep_nonzero
from .ops.multiply import get_mv_multiply
from .ops.add import get_mv_add
from .ops.simple_exp import get_mv_simple_exp
from .ops.select import get_mv_select
from .jaxga import reverse_indices, mv_repr
from .signatures import positive_signature
import jax.numpy as jnp


class MultiVector:
    def e(*indices, **kwargs):
        return MultiVector(
            values=jnp.ones([1], dtype=jnp.float32),
            indices=(tuple(indices),),
            signature=kwargs["signature"] if "signature" in kwargs else positive_signature
        )

    def __init__(self, values, indices, signature=positive_signature):
        self.values = values
        self.indices = tuple(indices)
        self.signature = signature

    def __add__(self, other):
        mv_add, out_indices = get_mv_add(self.indices, other.indices)
        out_values = mv_add(self.values, other.values)
        return MultiVector(values=out_values, indices=out_indices, signature=self.signature)

    def __sub__(self, other):
        mv_add, out_indices = get_mv_add(self.indices, other.indices)
        out_values = mv_add(self.values, -other.values)
        return MultiVector(values=out_values, indices=out_indices, signature=self.signature)

    def __mul__(self, other):
        if isinstance(other, MultiVector):
            mv_multiply, out_indices = get_mv_multiply(
                self.indices, other.indices, self.signature)
            out_values = mv_multiply(self.values, other.values)
            return MultiVector(values=out_values, indices=out_indices, signature=self.signature)
        return MultiVector(values=self.values * other, indices=self.indices, signature=self.signature)

    def __rmul__(self, other):
        if isinstance(other, MultiVector):
            mv_multiply, out_indices = get_mv_multiply(
                other.indices, self.indices, self.signature)
            out_values = mv_multiply(other.values, self.values)
            return MultiVector(values=out_values, indices=out_indices)
        return MultiVector(values=self.values * other, indices=self.indices, signature=self.signature)

    def __xor__(self, other):
        mv_multiply, out_indices = get_mv_multiply(
            self.indices, other.indices, self.signature, "op")
        out_values = mv_multiply(self.values, other.values)
        return MultiVector(values=out_values, indices=out_indices, signature=self.signature)

    def __or__(self, other):
        mv_multiply, out_indices = get_mv_multiply(
            self.indices, other.indices, self.signature, "ip")
        out_values = mv_multiply(self.values, other.values)
        return MultiVector(values=out_values, indices=out_indices, signature=self.signature)

    def __invert__(self):
        return MultiVector(values=self.values, indices=reverse_indices(self.indices), signature=self.signature)

    def __neg__(self):
        return MultiVector(values=-self.values, indices=reverse_indices(self.indices), signature=self.signature)

    def inverse(self):
        mv_inv, inv_indices = get_mv_inverse(self.indices, self.signature)
        inv_values = mv_inv(self.values)
        return MultiVector(values=inv_values, indices=inv_indices, signature=self.signature)

    def __truediv__(self, other):
        if isinstance(other, MultiVector):
            return self * other.inverse()
        return self * (1 / other)

    def __rtruediv__(self, other):
        return other * self.inverse()

    def __repr__(self):
        return mv_repr(self.indices, self.values)

    def __getitem__(self, select_indices):
        mv_select, out_indices = get_mv_select(self.indices, select_indices)
        out_values = mv_select(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def simple_exp(self):
        mv_simple_exp, out_indices = get_mv_simple_exp(
            self.indices, self.signature)
        out_values = mv_simple_exp(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def keep_nonzero(self):
        mv_keep_nonzero, out_indices = get_mv_keep_nonzero(
            self.indices, self.values
        )
        out_values = mv_keep_nonzero(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def reduce_same(self):
        mv_reduce_same, out_indices = get_mv_reduce_same(
            self.indices
        )
        out_values = mv_reduce_same(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def dual(self, dims):
        mv_dual, out_indices = get_mv_dual(self.indices, dims)
        out_values = mv_dual(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)
