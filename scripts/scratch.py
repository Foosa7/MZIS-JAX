import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
from src.engine import glynn_permanent, _factorial

@jit
def ryser_permanent(M):
    n = M.shape[0]
    num_subsets = (1 << n) - 1
    subset_indices = jnp.arange(1, num_subsets + 1)
    bit_positions = jnp.arange(n)
    masks = ((subset_indices[:, None] >> bit_positions[None, :]) & 1).astype(M.dtype)
    row_sums = masks @ M.T
    products = jnp.prod(row_sums, axis=1)
    popcounts = jnp.sum(masks.real.astype(jnp.int32), axis=1)
    signs = (-1.0) ** (popcounts + n)
    return jnp.sum(signs * products)

def test_closure():
    U = jnp.ones((8,8))
    in_indices = jnp.array([0,1,2])
    out_indices_arr = jnp.array([[0,1,2], [1,2,3], [2,3,4]])
    
    t0 = time.time()
    def compute_single_prob(out_idx):
        U_sub = U[out_idx][:, in_indices]
        return glynn_permanent(U_sub)
    vmap(compute_single_prob)(out_indices_arr).block_until_ready()
    return time.time() - t0

def test_no_closure():
    U = jnp.ones((8,8))
    in_indices = jnp.array([0,1,2])
    out_indices_arr = jnp.array([[0,1,2], [1,2,3], [2,3,4]])
    
    t0 = time.time()
    all_U_sub = U[out_indices_arr][:, :, in_indices]
    vmap(glynn_permanent)(all_U_sub).block_until_ready()
    return time.time() - t0

print("Closure 1:", test_closure())
print("Closure 2:", test_closure())
print("No Closure 1:", test_no_closure())
print("No Closure 2:", test_no_closure())
