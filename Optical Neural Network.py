import numpy as np
from scipy.optimize import minimize, root_scalar

# -------------------------------------------------
# Problem setup: NAND truth table
# -------------------------------------------------
X_data = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=float)

y_target = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)

# -------------------------------------------------
# Saturable absorber parameter
# -------------------------------------------------
T0 = 0.7      # unsaturated transmittance
EPS = 1e-12

# -------------------------------------------------
# 3 dB coupler matrix
# -------------------------------------------------
C = (1 / np.sqrt(2)) * np.array([
    [1,   -1j],
    [-1j,  1 ]
], dtype=complex)

# -------------------------------------------------
# Solve Shen saturable-absorber equation:
# x = 0.5 * ln(Tm / T0) / (1 - Tm), x >= 0
# Then use field-domain activation:
# Eout = sqrt(Tm) * Ein
# -------------------------------------------------
def solve_Tm(x, T0=T0):
    if x <= 0:
        return T0

    f = lambda t: 0.5 * np.log(t / T0) / (1 - t) - x
    sol = root_scalar(f, bracket=[T0 + EPS, 1 - EPS], method='brentq')
    return sol.root

def saturable_absorber_field(Ein, T0=T0):
    x = np.abs(Ein)**2
    Tm = solve_Tm(x, T0)
    return np.sqrt(Tm) * Ein

# -------------------------------------------------
# Forward model
# params = [theta1, theta2, phi1, phi2, b]
# -------------------------------------------------
def forward(params, X):
    theta1, theta2, phi1, phi2, b = params

    D_theta = np.diag([
        np.exp(-1j * theta1),
        np.exp(-1j * theta2)
    ])

    D_phi = np.diag([
        np.exp(-1j * phi1),
        np.exp(-1j * phi2)
    ])

    y_intensity = []
    y_field_all = []
    z_lin_all = []

    for x1, x2 in X:
        Xin = np.array([x1, x2], dtype=complex)

        # Z' = C * D_theta * C * X
        Zp = C @ D_theta @ C @ Xin

        # Z = D_phi * Z'
        Z = D_phi @ Zp

        # coherent sum + bias
        z_lin = Z[0] + Z[1] + b

        # field-domain saturable absorber
        y_field = saturable_absorber_field(z_lin, T0)

        y_field_all.append(y_field)
        z_lin_all.append(z_lin)

        # compare output intensity to NAND target
        y_intensity.append(np.abs(y_field)**2)

    y_intensity = np.array(y_intensity)
    y_field_all = np.array(y_field_all)
    z_lin_all = np.array(z_lin_all)

    # normalize intensities to [0,1] across the 4 truth-table points
    y_norm = y_intensity / (np.max(y_intensity) + EPS)

    return y_norm, y_field_all, z_lin_all

# -------------------------------------------------
# Loss function
# -------------------------------------------------
def loss_fn(params):
    y_pred, _, _ = forward(params, X_data)

    mse = np.mean((y_pred - y_target)**2)

    # small regularization on bias only
    reg = 1e-4 * params[4]**2

    return mse + reg

# -------------------------------------------------
# Practical bias snapping
# Example:
# 0.97 -> 1.0
# 0.55 -> 0.5
# -------------------------------------------------
def snap_bias_to_practical_value(b, step=0.5):
    return np.round(b / step) * step

# -------------------------------------------------
# Multi-start optimization
# -------------------------------------------------
best_result = None
best_loss = np.inf

for seed in range(20):
    rng = np.random.default_rng(seed)

    init = np.array([
        rng.uniform(0, 2*np.pi),   # theta1
        rng.uniform(0, 2*np.pi),   # theta2
        rng.uniform(0, 2*np.pi),   # phi1
        rng.uniform(0, 2*np.pi),   # phi2
        rng.uniform(-2, 2)         # bias
    ], dtype=float)

    result = minimize(
        loss_fn,
        init,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-9,
            'fatol': 1e-9,
            'disp': False
        }
    )

    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result

# -------------------------------------------------
# Final results
# -------------------------------------------------
params = best_result.x.copy()

# wrap phases into [0, 2pi)
params[:4] = np.mod(params[:4], 2*np.pi)

# save trained b before snapping
b_trained = params[4]

# snap b to a practical real optical field value
params[4] = snap_bias_to_practical_value(params[4], step=0.5)

theta1, theta2, phi1, phi2, b = params

# recompute outputs using the snapped practical b
y_pred, y_field, z_lin = forward(params, X_data)
hard_decision = (y_pred > 0.5).astype(int)

print("Optimized values:")
print(f"theta1 = {theta1:.6f} rad")
print(f"theta2 = {theta2:.6f} rad")
print(f"phi1   = {phi1:.6f} rad")
print(f"phi2   = {phi2:.6f} rad")
print(f"b (trained)   = {b_trained:.6f}")
print(f"b (practical) = {b:.6f}")

print("\nBest loss before snapping:")
print(f"{best_loss:.12e}")

print("\nLinear field before activation (Z):")
for x, z in zip(X_data, z_lin):
    print(f"X = {x.astype(int)}   Z = {z.real:.6f} + j{z.imag:.6f}")

print("\nActivated field output Y = f(Z):")
for x, y in zip(X_data, y_field):
    print(f"X = {x.astype(int)}   Y = {y.real:.6f} + j{y.imag:.6f}")

print("\nNormalized intensity outputs used for NAND target:")
for x, yp, hd, yt in zip(X_data, y_pred, hard_decision, y_target):
    print(f"X = {x.astype(int)}   y_pred = {yp:.6f}   hard = {hd}   target = {int(yt)}")
