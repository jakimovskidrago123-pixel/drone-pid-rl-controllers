"""Lab-ready version of the original PID-style position controller.

This file keeps the original PID controller method separate from controller.py.
It contains only controller-related code plus CSV logging, so it can be used in
the lab without the plotting/evaluation code from the original file.

Controller idea:
1. Compute position and yaw error.
2. Rotate horizontal position error into the drone yaw/body frame.
3. Use PID-style terms to command forward, lateral, vertical, and yaw velocity.
4. Saturate commands and use simple anti-windup when limits are reached.
"""

import csv
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Controller State
# ---------------------------------------------------------------------------
# The simulator repeatedly calls controller(...), so PID memory is stored at
# module level. reset_controller_state() clears this memory between runs.

int_x = 0.0
int_y = 0.0
int_z = 0.0
int_yaw = 0.0

prev_ex_b = 0.0
prev_ey_b = 0.0
prev_ez = 0.0
prev_eyaw = 0.0

prev_x_b = 0.0
prev_y_b = 0.0
prev_z = 0.0


# ---------------------------------------------------------------------------
# Original PID Gains And Limits
# ---------------------------------------------------------------------------
# These are embedded directly so the lab file does not depend on pid_gains.json.
GAINS = {
    "int_xy_limit": 5.0,
    "int_z_limit": 5.0,
    "int_yaw_limit": 2.0,
    "max_vxy": 1.0,
    "max_vz": 1.0,
    "max_yaw_rate": 5.0,
    "kp_xy": 0.7,
    "ki_xy": 0.0035,
    "kd_xy": 0.12,
    "kp_z": 1.60,
    "ki_z": 0.004,
    "kd_z": 0.15,
    "kp_yaw": 1.20,
    "ki_yaw": 0.0,
    "kd_yaw": 0.0,
}


# ---------------------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------------------
# The log file is created lazily on the first controller call. It starts fresh
# each time this module is imported in a new experiment process.
log_initialized = False
LOG_FILE = Path(__file__).with_name("controller_pidlab_log.csv")


def clamp(value, lower, upper):
    """Clamp value to the inclusive range [lower, upper]."""
    return max(lower, min(upper, value))


def wrap_angle(angle):
    """Wrap an angle in radians to the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def reset_controller_state():
    """Reset PID memory between simulation runs or targets."""
    global int_x, int_y, int_z, int_yaw
    global prev_ex_b, prev_ey_b, prev_ez, prev_eyaw
    global prev_x_b, prev_y_b, prev_z

    int_x = 0.0
    int_y = 0.0
    int_z = 0.0
    int_yaw = 0.0

    prev_ex_b = 0.0
    prev_ey_b = 0.0
    prev_ez = 0.0
    prev_eyaw = 0.0

    prev_x_b = 0.0
    prev_y_b = 0.0
    prev_z = 0.0


def log_controller_row(row):
    """Append one timestep of controller data to controller_pidlab_log.csv."""
    global log_initialized

    fieldnames = [
        "target_x",
        "target_y",
        "target_z",
        "target_yaw",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "error_x_world",
        "error_y_world",
        "error_z",
        "error_x_body",
        "error_y_body",
        "error_yaw",
        "pos_error_norm",
        "integral_x_world",
        "integral_y_world",
        "integral_z",
        "integral_yaw",
        "dpos_x_body",
        "dpos_y_body",
        "dpos_z",
        "dyaw_error",
        "vx_cmd",
        "vy_cmd",
        "vz_cmd",
        "yaw_rate_cmd",
    ]

    mode = "a" if log_initialized else "w"
    with LOG_FILE.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not log_initialized:
            writer.writeheader()
            log_initialized = True
        writer.writerow(row)


def controller(state, target_pos, dt, wind_enabled=False):
    """Compute velocity commands using the original PID-style controller.

    Args:
        state: Current drone state as (x, y, z, roll, pitch, yaw).
        target_pos: Target as (x_target, y_target, z_target, yaw_target).
        dt: Control timestep in seconds.
        wind_enabled: Present for compatibility with the assignment API.

    Returns:
        (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
    """
    global int_x, int_y, int_z, int_yaw
    global prev_ex_b, prev_ey_b, prev_ez, prev_eyaw
    global prev_x_b, prev_y_b, prev_z

    if dt <= 1e-6:
        dt = 1e-3

    x, y, z, roll, pitch, yaw = state
    x_d, y_d, z_d, yaw_d = target_pos

    # World-frame position error.
    ex_w = x_d - x
    ey_w = y_d - y
    ez = z_d - z
    eyaw = wrap_angle(yaw_d - yaw)

    # Rotate horizontal error into the drone yaw/body frame. The simulator's
    # lower-level controller expects forward/lateral velocity commands relative
    # to the current yaw heading.
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    ex_b = cy * ex_w + sy * ey_w
    ey_b = -sy * ex_w + cy * ey_w

    # Integral terms reduce steady-state position/yaw error. They are clamped to
    # prevent excessive buildup.
    int_x += ex_w * dt
    int_y += ey_w * dt
    int_z += ez * dt
    int_yaw += eyaw * dt

    int_x = clamp(int_x, -GAINS["int_xy_limit"], GAINS["int_xy_limit"])
    int_y = clamp(int_y, -GAINS["int_xy_limit"], GAINS["int_xy_limit"])
    int_z = clamp(int_z, -GAINS["int_z_limit"], GAINS["int_z_limit"])
    int_yaw = clamp(int_yaw, -GAINS["int_yaw_limit"], GAINS["int_yaw_limit"])

    # Rotate horizontal integral into body frame for the velocity command.
    int_xb = cy * int_x + sy * int_y
    int_yb = -sy * int_x + cy * int_y

    # Estimate body-frame position rate from finite differences. This is used
    # as the derivative/damping part of the PID-style controller.
    cur_x_b = cy * x + sy * y
    cur_y_b = -sy * x + cy * y

    dpx_b = (cur_x_b - prev_x_b) / dt
    dpy_b = (cur_y_b - prev_y_b) / dt
    dpz = (z - prev_z) / dt
    deyaw = wrap_angle(eyaw - prev_eyaw) / dt

    # PID-style velocity command. For position, the derivative term is applied
    # to measured position rate, which damps motion toward/through the target.
    vx_cmd = (
        GAINS["kp_xy"] * ex_b
        + GAINS["ki_xy"] * int_xb
        - GAINS["kd_xy"] * dpx_b
    )
    vy_cmd = (
        GAINS["kp_xy"] * ey_b
        + GAINS["ki_xy"] * int_yb
        - GAINS["kd_xy"] * dpy_b
    )
    vz_cmd = GAINS["kp_z"] * ez + GAINS["ki_z"] * int_z - GAINS["kd_z"] * dpz
    yaw_rate_cmd = (
        GAINS["kp_yaw"] * eyaw
        + GAINS["ki_yaw"] * int_yaw
        + GAINS["kd_yaw"] * deyaw
    )

    # Anti-windup: if the command is already saturating, undo this timestep's
    # integral contribution so the integrator does not keep growing.
    xy_norm_before = math.hypot(vx_cmd, vy_cmd)
    if xy_norm_before > GAINS["max_vxy"]:
        int_x -= ex_w * dt
        int_y -= ey_w * dt

    if abs(vz_cmd) >= GAINS["max_vz"]:
        int_z -= ez * dt

    if abs(yaw_rate_cmd) >= GAINS["max_yaw_rate"]:
        int_yaw -= eyaw * dt

    # Final command saturation.
    xy_norm = math.hypot(vx_cmd, vy_cmd)
    if xy_norm > GAINS["max_vxy"] and xy_norm > 1e-9:
        scale = GAINS["max_vxy"] / xy_norm
        vx_cmd *= scale
        vy_cmd *= scale

    vz_cmd = clamp(vz_cmd, -GAINS["max_vz"], GAINS["max_vz"])
    yaw_rate_cmd = clamp(
        yaw_rate_cmd, -GAINS["max_yaw_rate"], GAINS["max_yaw_rate"]
    )

    log_controller_row(
        {
            "target_x": x_d,
            "target_y": y_d,
            "target_z": z_d,
            "target_yaw": yaw_d,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "error_x_world": ex_w,
            "error_y_world": ey_w,
            "error_z": ez,
            "error_x_body": ex_b,
            "error_y_body": ey_b,
            "error_yaw": eyaw,
            "pos_error_norm": math.sqrt(ex_w * ex_w + ey_w * ey_w + ez * ez),
            "integral_x_world": int_x,
            "integral_y_world": int_y,
            "integral_z": int_z,
            "integral_yaw": int_yaw,
            "dpos_x_body": dpx_b,
            "dpos_y_body": dpy_b,
            "dpos_z": dpz,
            "dyaw_error": deyaw,
            "vx_cmd": vx_cmd,
            "vy_cmd": vy_cmd,
            "vz_cmd": vz_cmd,
            "yaw_rate_cmd": yaw_rate_cmd,
        }
    )

    prev_ex_b = ex_b
    prev_ey_b = ey_b
    prev_ez = ez
    prev_eyaw = eyaw

    prev_x_b = cur_x_b
    prev_y_b = cur_y_b
    prev_z = z

    return (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
