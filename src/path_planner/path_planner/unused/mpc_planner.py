import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, vertcat, Function, nlpsol

# Vehicle parameters
L = 2.9  # Wheelbase [m]
dt = 0.1  # Time step [s]
N = 10  # Prediction horizon

# MPC weights
Q = np.diag([1.0, 1.0, 0.5])  # [x, y, yaw] error. 3 *3 diagonal matrix -> Accuracy
R = np.diag([0.1, 0.1])  # [steer, accel]. 2 * 2 diagonal matrix -> Smoothness

# Limits
delta_max = np.radians(30)
a_max = 2.0


# Reference path
def generate_reference_path():
    t = np.linspace(0, 100, 201)
    x_ref = t
    y_ref = 2 * np.sin(0.2 * t)
    yaw_ref = np.arctan2(np.gradient(y_ref), np.gradient(x_ref))
    return x_ref, y_ref, yaw_ref


x_ref, y_ref, yaw_ref = generate_reference_path()


def kinematic_model(x, u):
    x_ = x[0]
    y_ = x[1]
    yaw = x[2]
    v = x[3]
    delta = u[0]
    a = u[1]

    x_next = x_ + v * np.cos(yaw) * dt
    y_next = y_ + v * np.sin(yaw) * dt
    yaw_next = yaw + v / L * np.tan(delta) * dt
    v_next = v + a * dt

    return vertcat(x_next, y_next, yaw_next, v_next)


# CasADi symbolic definition
x: SX = SX.sym("x")
y: SX = SX.sym("y")
yaw: SX = SX.sym("yaw")
v: SX = SX.sym("v")
delta: SX = SX.sym("delta")
a: SX = SX.sym("a")

state = vertcat(x, y, yaw, v)
control = vertcat(delta, a)
next_state = kinematic_model(state, control)

f = Function("f", [state, control], [next_state])

# MPC solver setup
from casadi import DM

x0 = DM([0.0, 0.0, 0.0, 0.0])
x0_full: np.ndarray = x0.full()
x_history = [x0_full.flatten()]

for t_step in range(100):
    # Extract ref over horizon
    start_idx = min(t_step, len(x_ref) - N)
    ref_traj = np.vstack(
        (
            x_ref[start_idx : start_idx + N],
            y_ref[start_idx : start_idx + N],
            yaw_ref[start_idx : start_idx + N],
        )
    )

    # Optimization variables
    X: SX = SX.sym("X", 4, N + 1)
    U: SX = SX.sym("U", 2, N)

    cost = 0
    constraints = []
    constraints.append(X[:, 0] - x0)

    for k in range(N):
        xk = X[:, k]  # State at time k
        uk = U[:, k]  # Control at time k
        xk_ref = ref_traj[:, k]  # Reference state at time k

        # Cost: tracking error + input
        state_err = xk[0:3] - xk_ref  # [x, y, yaw]
        cost += (
            state_err.T @ Q @ state_err + uk.T @ R @ uk
        )  # (1 * 3) @ (3 * 3) @ (3 * 1) + (2 * 1) @ (2 * 2) @ (2 * 1)

        # Dynamics constraint
        x_next = f(xk, uk)  # Function call for next state
        constraints.append(X[:, k + 1] - x_next)

    # Flatten constraints
    g = vertcat(*constraints)

    # NLP setup
    opt_vars = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
    nlp = {"x": opt_vars, "f": cost, "g": g}
    solver = nlpsol("solver", "ipopt", nlp)

    # Bounds
    lbx = []
    ubx = []
    for _ in range(N + 1):
        lbx += [-np.inf] * 4
        ubx += [np.inf] * 4
    for _ in range(N):
        lbx += [-delta_max, -a_max]
        ubx += [delta_max, a_max]

    lbg = [0] * ((N + 1) * 4)
    ubg = [0] * ((N + 1) * 4)

    # Initial guess
    x_init = np.tile(x0.full(), (1, N + 1))
    u_init = np.zeros((2, N))
    x_guess = np.concatenate((x_init.flatten(), u_init.flatten()))

    sol = solver(x0=x_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    u0 = sol["x"][4 * (N + 1) : 4 * (N + 1) + 2]

    # Apply control
    delta_opt = float(u0[0])
    a_opt = float(u0[1])
    x0 = f(x0, DM([delta_opt, a_opt]))
    x_history.append(x0.full().flatten())

# Plot result
x_history = np.array(x_history)
plt.figure()
plt.plot(x_ref, y_ref, "r--", label="Reference Path")
plt.plot(x_history[:, 0], x_history[:, 1], "b-", label="MPC Track")
plt.axis("equal")
plt.legend()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("MPC Path Tracking")
plt.grid(True)
plt.show()
