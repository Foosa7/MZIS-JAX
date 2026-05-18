import jax
import jax.numpy as jnp
import optax
import numpy as np

# Simulate a 2x2 unitary
def get_U(theta, phi):
    # A simple MZI
    c = jnp.cos(theta/2)
    s = jnp.sin(theta/2)
    ei_phi = jnp.exp(1j * phi)
    return jnp.array([
        [ei_phi * c, -s],
        [ei_phi * s, c]
    ])

# Target U - A 50/50 beam splitter but with some arbitrary output phases
U_target = jnp.array([
    [1j/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), 1j/np.sqrt(2)]
])

# Approach 1: User's approach (no alphas)
def loss_fn_1(params):
    theta, phi = params
    U_chip = get_U(theta, phi)
    return jnp.mean(jnp.abs(U_chip - U_target)**2)

# Approach 2: Existing approach (with alphas)
def loss_fn_2(params):
    theta, phi, alpha0, alpha1 = params
    U_chip = get_U(theta, phi)
    D = jnp.diag(jnp.array([jnp.exp(1j*alpha0), jnp.exp(1j*alpha1)]))
    U_full = D @ U_chip
    return jnp.mean(jnp.abs(U_full - U_target)**2)

# --- Optimize Approach 1 ---
optimizer1 = optax.adam(0.01)
opt_state1 = optimizer1.init(jnp.array([0.0, 0.0]))
params1 = jnp.array([0.0, 0.0])

@jax.jit
def step1(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn_1)(params)
    updates, opt_state = optimizer1.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for _ in range(2000):
    params1, opt_state1, loss1 = step1(params1, opt_state1)

# --- Optimize Approach 2 ---
optimizer2 = optax.adam(0.01)
opt_state2 = optimizer2.init(jnp.array([0.0, 0.0, 0.0, 0.0]))
params2 = jnp.array([0.0, 0.0, 0.0, 0.0])

@jax.jit
def step2(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn_2)(params)
    updates, opt_state = optimizer2.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for _ in range(2000):
    params2, opt_state2, loss2 = step2(params2, opt_state2)


print("=== Without Virtual Alphas (Direct Frobenius) ===")
print("Final Loss:", loss1)
U_chip_1 = get_U(params1[0], params1[1])
print("Target Probabilities (|U|^2):\n", np.abs(U_target)**2)
print("Result Probabilities (|U|^2):\n", np.abs(U_chip_1)**2)

print("\n=== With Virtual Alphas (Existing approach) ===")
print("Final Loss:", loss2)
U_chip_2 = get_U(params2[0], params2[1])
print("Target Probabilities (|U|^2):\n", np.abs(U_target)**2)
print("Result Probabilities (|U|^2):\n", np.abs(U_chip_2)**2)
