import numpy as np
import sympy as sp

# ============================================================================
# Trigonometric utilities (from trigon.py)
# ============================================================================

def atan2f(y, x, tolerance=1e-6, to_degree=False):
    zero_y = np.abs(y) <= tolerance
    zero_x = np.abs(x) <= tolerance
    if zero_x and zero_y:
        rad = 0
    elif zero_x and (not zero_y):
        rad = np.pi/2 if y > tolerance else -np.pi/2
    elif (not zero_x) and zero_y:
        rad = 0 if x > tolerance else np.pi
    else:
        rad = np.arctan2(y, x)
    if to_degree:
        return np.rad2deg(rad)
    else:
        return rad
    
def angle_diff(comp_src, comp_dst, offset=0, tolerance=1e-6, wrap=True, to_degree=False):
    zero_src = np.abs(comp_src) <= tolerance
    zero_dst = np.abs(comp_dst) <= tolerance
    if zero_src and zero_dst:
        rad = 0
    elif zero_src and (not zero_dst):
        rad = np.angle(comp_dst)
    elif (not zero_src) and zero_dst:
        rad = -np.angle(comp_src)
    else:
        rad = np.angle(comp_dst) - np.angle(comp_src)
    rad += offset
    if wrap:
        rad = np.mod(rad, 2 * np.pi)
    if to_degree:
        return np.rad2deg(rad)
    else:
        return rad

# ============================================================================
# Unitary block transformations (from ublock.py)
# ============================================================================

def U2BS(dim, m, n, phi, theta, use_sym=False, Lp=1, Lc=1):
    assert m < n < dim
    if use_sym:
        mat = sp.eye(dim)
        mat[m, m] = sp.sqrt(Lp) * sp.exp(sp.I * phi) * sp.cos(theta)
        mat[m, n] = sp.sqrt(Lc) * sp.I * sp.sin(theta)
        mat[n, m] = sp.sqrt(Lc) * sp.I * sp.exp(sp.I * phi) * sp.sin(theta)
        mat[n, n] = sp.sqrt(Lp) * sp.cos(theta)
    else:
        mat = np.eye(dim, dtype=np.complex128)
        mat[m, m] = np.sqrt(Lp) * np.exp(1j * phi) * np.cos(theta)
        mat[m, n] = np.sqrt(Lc) * 1j * np.sin(theta)
        mat[n, m] = np.sqrt(Lc) * 1j * np.exp(1j * phi) * np.sin(theta)
        mat[n, n] = np.sqrt(Lp) * np.cos(theta)
    return mat

def U2MZI(dim, m, n, phi, theta, use_sym=False, Lp=1, Lc=1):
    assert m < n < dim
    if use_sym:
        mat = sp.eye(dim)
        mat[m, m] = sp.sqrt(Lp) * sp.I * sp.exp(sp.I * phi) * sp.sin(theta)
        mat[m, n] = sp.sqrt(Lc) * sp.I * sp.cos(theta)
        mat[n, m] = sp.sqrt(Lc) * sp.I * sp.exp(sp.I * phi) * sp.cos(theta)
        mat[n, n] = -sp.sqrt(Lp) * sp.I * sp.sin(theta)
    else:
        mat = np.eye(dim, dtype=np.complex128)
        mat[m, m] = np.sqrt(Lp) * 1j * np.exp(1j * phi) * np.sin(theta)
        mat[m, n] = np.sqrt(Lc) * 1j * np.cos(theta)
        mat[n, m] = np.sqrt(Lc) * 1j * np.exp(1j * phi) * np.cos(theta)
        mat[n, n] = -np.sqrt(Lp) * 1j * np.sin(theta)
    return mat

# ============================================================================
# Clements decomposition (from clements.py)
# ============================================================================

def decompose_clements(u, block='bs'):
    assert isinstance(u, np.ndarray)
    assert isinstance(block, str) and block.strip().lower() in ['bs', 'mzi']
    if len(u.shape) != 2:
        raise ValueError("U(N) should be 2-dimension matrix.")
        
    if u.shape[0] != u.shape[1]:
        raise ValueError("U(N) should be a square matrix.")
        
    mat = u.copy().astype(np.complex128)
    dim = mat.shape[0]
    
    row = dim - 1
    col = int(np.ceil(dim / 2))
    
    cnt_fore = np.zeros(row, dtype=int)
    cnt_back = np.ones(row, dtype=int) * (col - 1)
    if dim % 2 == 1:
        cnt_back[1::2] = col - 2
    
    phis = np.zeros((row, col))
    thetas = np.zeros((row, col))
    alphas = np.zeros(dim)
    
    for p in range(dim-1):
        for q in range(p+1):
            if p % 2 == 0:
                x = dim - 1 - q
                y = p - q
                if block == 'bs':
                    theta = atan2f(np.abs(mat[x,y]), np.abs(mat[x,y+1]))
                    phi = angle_diff(mat[x,y+1], mat[x,y], offset=-np.pi/2)
                    U2block = U2BS
                elif block == 'mzi':
                    theta = np.pi/2 - atan2f(np.abs(mat[x,y]), np.abs(mat[x,y+1]))
                    phi = angle_diff(mat[x,y+1], mat[x,y], offset=np.pi)
                    U2block = U2MZI
                mat = mat @ U2block(dim, y, y+1, phi, theta).conj().T
                thetas[y, cnt_fore[y]] = theta
                phis[y, cnt_fore[y]] = phi
                cnt_fore[y] += 1
            else:
                x = dim - 1 - p + q
                y = q
                if block == 'bs':
                    theta = atan2f(np.abs(mat[x,y]), np.abs(mat[x-1,y]))
                    phi = angle_diff(mat[x-1,y], mat[x,y], offset=np.pi/2)
                    U2block = U2BS
                elif block == 'mzi':
                    theta = np.pi/2 - atan2f(np.abs(mat[x,y]), np.abs(mat[x-1,y]))
                    phi = angle_diff(mat[x-1,y], mat[x,y], offset=0)
                    U2block = U2MZI
                mat = U2block(dim, x-1, x, phi, theta) @ mat
                thetas[x-1, cnt_back[x-1]] = theta
                phis[x-1, cnt_back[x-1]] = phi
                cnt_back[x-1] -= 1
    for p in range(dim-2, -1, -1):
        for q in range(p, -1, -1):
            if p % 2 == 0:
                continue
            x = dim - 1 - p + q
            y = q
            cnt_back[x-1] += 1
            theta = thetas[x-1, cnt_back[x-1]]
            phi = phis[x-1, cnt_back[x-1]]
            eta1 = mat[x-1, x-1]
            eta2 = mat[x, x]
            if block == 'bs':
                phi_new = angle_diff(eta2, -eta1, offset=0)
                mat[x-1, x-1] = eta1 * np.exp(-1j * (phi+phi_new))
            elif block == 'mzi':
                phi_new = angle_diff(eta2, eta1, offset=0)
                mat[x-1, x-1] = -eta1 * np.exp(-1j * (phi+phi_new))
                mat[x, x] = -eta2
            phis[x-1, cnt_back[x-1]] = phi_new
    for i in range(dim):
        alphas[i] = np.angle(mat[i, i])
    return phis, thetas, alphas

def reconstruct_clements(phis, thetas, alphas, block='bs', Lp_dB=0, Lc_dB=0):
    assert len(phis.squeeze().shape) == 2 or phis.size == 1
    assert len(thetas.squeeze().shape) == 2 or thetas.size == 1
    assert len(alphas.squeeze().shape) == 1
    assert phis.squeeze().shape == thetas.squeeze().shape
    assert isinstance(block, str) and block.strip().lower() in ['bs', 'mzi']
    
    if block == 'bs':
        U2block = U2BS
    elif block == 'mzi':
        U2block = U2MZI
    
    if thetas.size == 1:
        row = 1
        col = 1
    else:
        row, col = thetas.squeeze().shape
    dim = row + 1
    num = int(dim * (dim - 1) / 2) 
    assert alphas.squeeze().shape[0] == dim
    
    Lp = 10 ** (Lp_dB / 10)
    Lc = 10 ** (Lc_dB / 10)
    
    sft = np.diag(np.exp(1j * alphas))
    mat = np.eye(dim)
    for p in range(col):
        for q in range(0, row, 2):
            mat = U2block(dim, q, q+1, phis[q,p], thetas[q,p], Lp=Lp, Lc=Lc) @ mat
        if p >= col - 1 and dim % 2 == 1:
            continue
        for q in range(1, row, 2):
            mat = U2block(dim, q, q+1, phis[q,p], thetas[q,p], Lp=Lp, Lc=Lc) @ mat
    mat = sft @ mat
    return mat

# ============================================================================
# Mapping into the simulation engine's MZI convention
# ============================================================================

def clements_to_engine_phases(phis, thetas, alphas, layout):
    """Converts Clements parameters into phase settings for the JAX Engine mesh.

    The two conventions are not interchangeable. `U2MZI` puts the internal phase
    symmetrically on both arms, while the Engine (like a real chip) drives a
    single arm, so its block carries an extra common-mode factor:

        engine_block(t, p) = exp(i*t/2) * U2MZI(pi + t/2, p - pi)

    Naively setting `theta_engine = 2*theta_pnn` therefore leaves a per-MZI
    phase error that does not cancel across the mesh (~0.33 absolute error in
    the transfer matrix for a random 8x8 unitary). Permutations happen to
    survive it, because every MZI sits at full bar or cross where the stray
    phases move no power -- which is why the switching demos looked correct.

    The common-mode terms are pushed forward through the mesh instead. Carrying
    an accumulated input-phase vector `d` and matching

        E(t_e, p_e) @ diag(e^{i d_t}, e^{i d_t+1})
            == diag(e^{i g_t}, e^{i g_t+1}) @ U2MZI(t_p, p_p)

    determines every unknown uniquely:

        t_e = 2*(t_p - pi)
        p_e = p_p + d[top+1] - d[top] + pi
        g   = t_p - pi + d[top+1]          (applied to both of the MZI's modes)

    The leftovers accumulate at the output, exactly as `grid_common_mode_flow`
    does in neurophox. A mesh of 2-port MZIs has no output phase shifters, so
    the result is exact up to that final diagonal -- |U| matches the target
    exactly, which is what determines routed power.

    Args:
        phis, thetas, alphas: output of `decompose_clements(U, block='mzi')`.
        layout: `Engine.layout` -- columns of {'id', 'mode_top'} dicts.

    Returns:
        settings: {mzi_id: (theta, phi)}, both wrapped into [0, 2*pi).
        output_phases: length-n_modes residual screen, such that
            U_engine == diag(exp(1j*output_phases)) @ U_target.
    """
    n_modes = len(alphas)
    carry = np.zeros(n_modes)
    settings = {}

    for col_idx, col in enumerate(layout):
        p = col_idx // 2
        # MZIs within a column act on disjoint mode pairs, so the whole column
        # reads the incoming carry and writes a fresh one.
        new_carry = carry.copy()
        for mzi in col:
            top = mzi['mode_top']
            theta_p, phi_p = thetas[top, p], phis[top, p]

            theta_e = 2.0 * (theta_p - np.pi)
            phi_e = phi_p + carry[top + 1] - carry[top] + np.pi
            new_carry[top] = new_carry[top + 1] = theta_p - np.pi + carry[top + 1]

            settings[mzi['id']] = (float(np.mod(theta_e, 2 * np.pi)),
                                   float(np.mod(phi_e, 2 * np.pi)))
        carry = new_carry

    return settings, np.mod(carry - alphas, 2 * np.pi)
    