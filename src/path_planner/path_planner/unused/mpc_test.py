import casadi as ca
import numpy as np
from matplotlib import pyplot as plt

# MPC horizon and timestep
N = 10
dt = 0.1

# State and control variables
nx = 2  # [x, y]
nu = 3  # [vx, vy, omega]

# symbolic variables
x = ca.MX.sym("x")
y = ca.MX.sym("y")
vx = ca.MX.sym("vx")
vy = ca.MX.sym("vy")
omega = ca.MX.sym("omega")

state = ca.vertcat(x, y)
control = ca.vertcat(vx, vy, omega)

# Kinematic model
dx = dt * (1.0 * vx + 0.2 * vy + 0.1 * omega)
dy = dt * (0.3 * vx + 1.0 * vy + 0.05 * omega)
next_state = ca.vertcat(x + dx, y + dy)

# CasADi function
f_model = ca.Function("f_model", [state, control], [next_state])

# Optimization variables
X = ca.MX.sym("X", nx, N + 1)
U = ca.MX.sym("U", nu, N)

# Reference path: sine wave
A = 0.2  # amplitude
f = 0.5  # frequency (Hz)
t_vals = np.linspace(0, N * dt, N + 1)

x_ref = t_vals
y_ref = A * np.sin(2 * np.pi * f * t_vals)
ref = np.vstack((x_ref, y_ref))


# Cost weights
Q = np.diag([10.0, 10.0])
R = np.diag([1.0, 1.0, 0.1])

# Cost and constraints
cost = 0
g = []

for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    x_next = f_model(xk, uk)
    g.append(X[:, k + 1] - x_next)

    cost += ca.mtimes([(xk - ref[:, k]).T, Q, (xk - ref[:, k])]) + ca.mtimes(
        [uk.T, R, uk]
    )

# Terminal cost
cost += ca.mtimes([(X[:, N] - ref[:, N]).T, Q, (X[:, N] - ref[:, N])])
g = ca.vertcat(*g)

# Flatten decision variables
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

# Define problem
nlp = {"x": opt_vars, "f": cost, "g": g}
solver = ca.nlpsol("solver", "ipopt", nlp)

# Initial state
x0 = np.array([0.0, 0.0])
x_init = np.tile(x0, (N + 1, 1)).T
u_init = np.zeros((nu, N))

# Initial guess
x_guess = np.concatenate([x_init.flatten(), u_init.flatten()])
lbx = [-ca.inf] * len(x_guess)
ubx = [ca.inf] * len(x_guess)
lbg = [0.0] * (nx * N)
ubg = [0.0] * (nx * N)

# Solve
sol = solver(x0=x_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
opt = sol["x"].full().flatten()

# Parse result
X_opt = opt[: nx * (N + 1)].reshape((nx, N + 1))
U_opt = opt[nx * (N + 1) :].reshape((nu, N))

# Print
for t in range(N + 1):
    print(f"Step {t}: x = {X_opt[0, t]:.3f}, y = {X_opt[1, t]:.3f}")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_ref, y_ref, label="Reference Path", linestyle="--")
plt.plot(X_opt[0], X_opt[1], label="MPC Trajectory", marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.title("MPC Tracking of Sine Wave Reference Path")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
