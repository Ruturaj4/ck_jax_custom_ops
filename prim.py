# Author: Ruturaj

from functools import partial, reduce

import jax
import jaxlib
import jax.numpy as jnp
from jax import core, dtypes
from jax.lib import xla_client
from jax.core import ShapedArray
import jax._src.test_util as jtu
from jax.interpreters.mlir import ir
from jax.interpreters import xla, mlir
from jaxlib.hlo_helpers import custom_call

from build import gpu_ops

# Create _gemm_p for forward operation.
_gemm_p = core.Primitive("gemm_runner")
_gemm_p.multiple_results = False
_gemm_p.def_impl(partial(xla.apply_primitive, _gemm_p))

def gemm_runner(A, B):
    C = jax.lax.full((A.shape[0], B.shape[1]), 0.0, dtype=A.dtype)
    _gemm_p.bind(A, B, C)
    return C

####################
# Lowering to MLIR #
####################

# Register functions defined in gpu_ops as custom call target for GPUs.
for _name, _value in gpu_ops.get_gemm_registrations().items():
    # _name, _value -> gemm_runner; <capsule object "xla._CUSTOM_CALL_TARGET" at 0x7ff766e73d20>
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")

def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        # ir.F64Type.get(): gpu_ops.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _gemm_rocm_lowering(ctx, A, B, C):
    A_type = ir.RankedTensorType(A.type)
    B_type = ir.RankedTensorType(B.type)
    C_type = ir.RankedTensorType(C.type)
    A_shape = A_type.shape
    B_shape = B_type.shape
    C_shape = C_type.shape

    # Determine output shape based on matrix multiplication rules.
    out_shape = (A_shape[0], B_shape[1])
    # The outer shape doesn't change.
    out_type = B_type.element_type

    operand_layouts = default_layouts(A_shape, B_shape, C_shape)
    result_layouts = default_layouts(out_shape)

    # Setting up gemm configurations.
    opaque = gpu_ops.create_gemm_config_descriptor(
        32,
        32,
        32,
        element_type_to_descriptor_type_mapping(A_type.element_type),
    )

    out = custom_call(
        "gemm_runner",
        operands=[A, B, C],
        operand_layouts=operand_layouts,
        result_types=[ir.RankedTensorType.get(out_shape, out_type)],
        result_layouts=result_layouts,
        backend_config=opaque,
    ).results
    return out


mlir.register_lowering(
    _gemm_p,
    _gemm_rocm_lowering,
    platform="gpu",
)

#######################
# Abstract evaluation #
#######################

def _gemm_abstract(A, B, C):
    assert A.shape == B.shape, "Shapes of A and B must match."
    assert A.shape == C.shape, "Shapes of A and C must match."
    return ShapedArray(A.shape, A.dtype)

_gemm_p.def_abstract_eval(_gemm_abstract)

A = jnp.ones((32, 32), dtype=jnp.float32)
B = jnp.ones((32, 32), dtype=jnp.float32)

print(A)
print(B)
C = gemm_runner(A, B)
print(C)
