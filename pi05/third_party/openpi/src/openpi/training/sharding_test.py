import concurrent.futures
import threading

import jax
import jax.numpy as jnp

from openpi.training import sharding


def _constrains() -> bool:
    # A fresh function each call, so jit's trace cache cannot answer for an earlier mesh state.
    jaxpr = jax.make_jaxpr(lambda x: sharding.activation_sharding_constraint(x))(jnp.zeros((2,)))
    return "sharding_constraint" in str(jaxpr)


def test_set_mesh_applies_only_on_its_own_thread():
    # The trainer lowers its train step under set_mesh on a startup thread while the main thread
    # builds the train state; neither may see or block on the other's mesh.
    mesh = sharding.make_mesh(1)
    entered, release = threading.Event(), threading.Event()

    def hold_mesh():
        with sharding.set_mesh(mesh):
            entered.set()
            release.wait(10)
            return _constrains()

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        held = pool.submit(hold_mesh)
        assert entered.wait(10)
        assert not _constrains()
        with sharding.set_mesh(mesh):
            assert _constrains()
        release.set()
        assert held.result()
    assert not _constrains()
