"""Reward-trained neural policy controller for the Tello simulator.

This is a real RL-style controller file: the action is produced by a small
neural policy loaded from rl_neural_policy_weights.json.  The policy was trained
with reward-based policy search in train_rl_neural_policy.py.

The simulator API is unchanged:

    controller(state, target_pos, dt, wind_enabled=False)

returns:

    (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)

For reliability, the neural network is a residual actor: a stable base policy
provides sensible commands, and the trained neural network adds a learned
correction. This is a common safe way to apply RL to control problems.
"""

import json
import math
from pathlib import Path


FEATURE_DIM = 12
HIDDEN_DIM = 12
ACTION_DIM = 4

WEIGHTS_FILE = Path(__file__).with_name("rl_neural_policy_weights.json")


prev_x_b = 0.0
prev_y_b = 0.0
prev_z = 0.0
prev_eyaw = 0.0
state_initialized = False

int_x = 0.0
int_y = 0.0
int_z = 0.0
int_yaw = 0.0

elapsed_time = 0.0
settled_time = 0.0
integral_enabled = False
last_target = None
policy_cache = None


LIMITS = {
    "max_vxy": 1.0,
    "max_vz": 1.0,
    "max_yaw_rate": 1.74533,
    "settle_error_gate": 0.12,
    "settle_vel_gate": 0.08,
    "settle_time_required": 0.6,
    "int_xy_limit": 3.0,
    "int_z_limit": 2.0,
    "int_yaw_limit": 1.0,
}


def default_policy():
    base_w = [[0.0 for _ in range(FEATURE_DIM)] for _ in range(ACTION_DIM)]
    base_b = [0.0 for _ in range(ACTION_DIM)]

    # Base actor seeded from the tuned cascaded controller:
    # [ex_b, ey_b, ez, eyaw, vx_b, vy_b, vz, yaw_rate, int_xb, int_yb, int_z, int_yaw]
    base_w[0][0] = 0.9064
    base_w[0][4] = -0.03
    base_w[0][8] = 0.0036
    base_w[1][1] = 0.9064
    base_w[1][5] = -0.03
    base_w[1][9] = 0.0036
    base_w[2][2] = 1.854
    base_w[2][6] = -0.03
    base_w[2][10] = 0.0041
    base_w[3][3] = 1.2

    return {
        "base_w": base_w,
        "base_b": base_b,
        "hidden_w": [[0.0 for _ in range(FEATURE_DIM)] for _ in range(HIDDEN_DIM)],
        "hidden_b": [0.0 for _ in range(HIDDEN_DIM)],
        "out_w": [[0.0 for _ in range(HIDDEN_DIM)] for _ in range(ACTION_DIM)],
        "out_b": [0.0 for _ in range(ACTION_DIM)],
    }


def load_policy():
    global policy_cache
    if policy_cache is not None:
        return policy_cache

    policy = default_policy()
    if WEIGHTS_FILE.exists():
        try:
            loaded = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
            policy.update(loaded["policy"])
        except Exception:
            # Fall back to the stable default policy if the weight file is not
            # available or is malformed.
            pass
    policy_cache = policy
    return policy_cache


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def dot(row, values):
    total = 0.0
    for weight, value in zip(row, values):
        total += weight * value
    return total


def neural_policy(obs):
    policy = load_policy()

    action = [
        dot(policy["base_w"][i], obs) + policy["base_b"][i]
        for i in range(ACTION_DIM)
    ]

    hidden = [
        math.tanh(dot(policy["hidden_w"][i], obs) + policy["hidden_b"][i])
        for i in range(HIDDEN_DIM)
    ]

    for i in range(ACTION_DIM):
        action[i] += dot(policy["out_w"][i], hidden) + policy["out_b"][i]

    return action


def reset_target_state():
    global prev_x_b, prev_y_b, prev_z, prev_eyaw, state_initialized
    global int_x, int_y, int_z, int_yaw
    global elapsed_time, settled_time, integral_enabled

    prev_x_b = 0.0
    prev_y_b = 0.0
    prev_z = 0.0
    prev_eyaw = 0.0
    state_initialized = False

    int_x = 0.0
    int_y = 0.0
    int_z = 0.0
    int_yaw = 0.0

    elapsed_time = 0.0
    settled_time = 0.0
    integral_enabled = False


def reset_controller_state():
    global last_target
    reset_target_state()
    last_target = None


def controller(state, target_pos, dt, wind_enabled=False):
    global prev_x_b, prev_y_b, prev_z, prev_eyaw, state_initialized
    global int_x, int_y, int_z, int_yaw
    global elapsed_time, settled_time, integral_enabled
    global last_target

    current_target = tuple(float(v) for v in target_pos)
    if last_target is None:
        last_target = current_target
    elif current_target != last_target:
        reset_target_state()
        last_target = current_target

    if dt <= 1e-6:
        dt = 1e-3
    elapsed_time += dt

    x, y, z, roll, pitch, yaw = state
    x_d, y_d, z_d, yaw_d = target_pos

    ex_w = x_d - x
    ey_w = y_d - y
    ez = z_d - z
    eyaw = wrap_angle(yaw_d - yaw)
    pos_error = math.sqrt(ex_w * ex_w + ey_w * ey_w + ez * ez)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    ex_b = cy * ex_w + sy * ey_w
    ey_b = -sy * ex_w + cy * ey_w

    cur_x_b = cy * x + sy * y
    cur_y_b = -sy * x + cy * y
    if not state_initialized:
        vel_x_b = 0.0
        vel_y_b = 0.0
        vel_z = 0.0
        yaw_error_rate = 0.0
        state_initialized = True
    else:
        vel_x_b = (cur_x_b - prev_x_b) / dt
        vel_y_b = (cur_y_b - prev_y_b) / dt
        vel_z = (z - prev_z) / dt
        yaw_error_rate = wrap_angle(eyaw - prev_eyaw) / dt

    speed_est = math.sqrt(vel_x_b * vel_x_b + vel_y_b * vel_y_b + vel_z * vel_z)
    if (
        pos_error <= LIMITS["settle_error_gate"]
        and speed_est <= LIMITS["settle_vel_gate"]
    ):
        settled_time += dt
        if settled_time >= LIMITS["settle_time_required"]:
            integral_enabled = True
    else:
        settled_time = 0.0

    if integral_enabled:
        int_x += ex_w * dt
        int_y += ey_w * dt
        int_z += ez * dt
        int_yaw += eyaw * dt
        int_x = clamp(int_x, -LIMITS["int_xy_limit"], LIMITS["int_xy_limit"])
        int_y = clamp(int_y, -LIMITS["int_xy_limit"], LIMITS["int_xy_limit"])
        int_z = clamp(int_z, -LIMITS["int_z_limit"], LIMITS["int_z_limit"])
        int_yaw = clamp(int_yaw, -LIMITS["int_yaw_limit"], LIMITS["int_yaw_limit"])
    else:
        int_x = 0.0
        int_y = 0.0
        int_z = 0.0
        int_yaw = 0.0

    int_xb = cy * int_x + sy * int_y
    int_yb = -sy * int_x + cy * int_y

    obs = [
        clamp(ex_b, -5.0, 5.0),
        clamp(ey_b, -5.0, 5.0),
        clamp(ez, -4.0, 4.0),
        clamp(eyaw, -math.pi, math.pi),
        clamp(vel_x_b, -3.0, 3.0),
        clamp(vel_y_b, -3.0, 3.0),
        clamp(vel_z, -3.0, 3.0),
        clamp(yaw_error_rate, -10.0, 10.0),
        clamp(int_xb, -LIMITS["int_xy_limit"], LIMITS["int_xy_limit"]),
        clamp(int_yb, -LIMITS["int_xy_limit"], LIMITS["int_xy_limit"]),
        clamp(int_z, -LIMITS["int_z_limit"], LIMITS["int_z_limit"]),
        clamp(int_yaw, -LIMITS["int_yaw_limit"], LIMITS["int_yaw_limit"]),
    ]

    vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = neural_policy(obs)

    xy_norm = math.hypot(vx_cmd, vy_cmd)
    if xy_norm > LIMITS["max_vxy"] and xy_norm > 1e-9:
        scale = LIMITS["max_vxy"] / xy_norm
        vx_cmd *= scale
        vy_cmd *= scale

    vz_cmd = clamp(vz_cmd, -LIMITS["max_vz"], LIMITS["max_vz"])
    yaw_rate_cmd = clamp(
        yaw_rate_cmd,
        -LIMITS["max_yaw_rate"],
        LIMITS["max_yaw_rate"],
    )

    prev_x_b = cur_x_b
    prev_y_b = cur_y_b
    prev_z = z
    prev_eyaw = eyaw

    return (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
