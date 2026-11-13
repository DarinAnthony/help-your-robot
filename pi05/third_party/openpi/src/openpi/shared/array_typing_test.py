from flax import struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@at.typecheck
@struct.dataclass
class _State:
    step: at.Int[at.ArrayLike, ""]
    value: at.Float[at.ArrayLike, "n"]


def test_ahead_of_time_lowering_of_typechecked_dataclass():
    # The trainer compiles its train step for an abstract TrainState on a startup thread; lowering
    # rebuilds the dataclass from ArgInfo placeholders, which must not be typechecked.
    state = _State(step=jnp.asarray(2, jnp.int32), value=jnp.ones(3))
    abstract = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), state)
    compiled = jax.jit(lambda s: s.value * s.step).lower(abstract).compile()
    assert compiled.args_info[0][0].step.shape == ()
    assert jnp.array_equal(compiled(state), jnp.full(3, 2.0))
