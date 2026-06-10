"""Train a neural residual policy with reward-based policy search.

This script trains rl_neural_policy_weights.json for controller_rl_neural.py.
It uses a lightweight first-order drone model so the policy can be trained in
seconds instead of requiring thousands of slow PyBullet rollouts.

Algorithm:
    Cross-Entropy Method / evolutionary policy search.

This is true reward-based RL: candidates are evaluated only by episode reward,
and the best candidates update the policy distribution.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import controller_rl_neural as actor


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_targets(path):
    targets = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 4:
                continue
            target = tuple(float(v) for v in row)
            if target[2] >= 0:
                targets.append(target)
    if not targets:
        raise RuntimeError(f"No targets found in {path}")
    return targets


def default_policy_np():
    policy = actor.default_policy()
    return {
        key: np.array(value, dtype=float)
        for key, value in policy.items()
    }


def residual_shapes():
    return {
        "hidden_w": (actor.HIDDEN_DIM, actor.FEATURE_DIM),
        "hidden_b": (actor.HIDDEN_DIM,),
        "out_w": (actor.ACTION_DIM, actor.HIDDEN_DIM),
        "out_b": (actor.ACTION_DIM,),
    }


def flatten_residual(policy):
    parts = []
    for key, shape in residual_shapes().items():
        parts.append(policy[key].reshape(-1))
    return np.concatenate(parts)


def unflatten_residual(vector, base_policy):
    policy = {key: np.array(value, dtype=float).copy() for key, value in base_policy.items()}
    cursor = 0
    for key, shape in residual_shapes().items():
        size = int(np.prod(shape))
        policy[key] = vector[cursor:cursor + size].reshape(shape)
        cursor += size
    return policy


def policy_action(policy, obs):
    base = policy["base_w"] @ obs + policy["base_b"]
    hidden = np.tanh(policy["hidden_w"] @ obs + policy["hidden_b"])
    action = base + policy["out_w"] @ hidden + policy["out_b"]

    xy_norm = np.linalg.norm(action[:2])
    if xy_norm > 1.0:
        action[:2] *= 1.0 / xy_norm
    action[2] = np.clip(action[2], -1.0, 1.0)
    action[3] = np.clip(action[3], -1.74533, 1.74533)
    return action


def rollout(policy, target, dt=0.05, total_time=20.0):
    x = 0.0
    y = 0.0
    z = 1.0
    yaw = 0.0
    vx_b = 0.0
    vy_b = 0.0
    vz = 0.0
    yaw_rate = 0.0

    int_x = 0.0
    int_y = 0.0
    int_z = 0.0
    int_yaw = 0.0
    settled_time = 0.0
    integral_enabled = False

    tau_v = 0.32
    tau_z = 0.28
    tau_yaw = 0.22
    steps = int(total_time / dt)
    last_half_start = steps // 2
    pos_errors = []
    yaw_errors = []
    action_costs = []

    for step in range(steps):
        x_d, y_d, z_d, yaw_d = target
        ex_w = x_d - x
        ey_w = y_d - y
        ez = z_d - z
        eyaw = wrap_angle(yaw_d - yaw)

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        ex_b = cy * ex_w + sy * ey_w
        ey_b = -sy * ex_w + cy * ey_w
        speed = math.sqrt(vx_b * vx_b + vy_b * vy_b + vz * vz)
        pos_error = math.sqrt(ex_w * ex_w + ey_w * ey_w + ez * ez)

        if pos_error <= 0.12 and speed <= 0.08:
            settled_time += dt
            if settled_time >= 0.6:
                integral_enabled = True
        else:
            settled_time = 0.0

        if integral_enabled:
            int_x = np.clip(int_x + ex_w * dt, -3.0, 3.0)
            int_y = np.clip(int_y + ey_w * dt, -3.0, 3.0)
            int_z = np.clip(int_z + ez * dt, -2.0, 2.0)
            int_yaw = np.clip(int_yaw + eyaw * dt, -1.0, 1.0)
        else:
            int_x = 0.0
            int_y = 0.0
            int_z = 0.0
            int_yaw = 0.0

        int_xb = cy * int_x + sy * int_y
        int_yb = -sy * int_x + cy * int_y

        obs = np.array(
            [
                np.clip(ex_b, -5.0, 5.0),
                np.clip(ey_b, -5.0, 5.0),
                np.clip(ez, -4.0, 4.0),
                np.clip(eyaw, -math.pi, math.pi),
                np.clip(vx_b, -3.0, 3.0),
                np.clip(vy_b, -3.0, 3.0),
                np.clip(vz, -3.0, 3.0),
                np.clip(-yaw_rate, -10.0, 10.0),
                int_xb,
                int_yb,
                int_z,
                int_yaw,
            ],
            dtype=float,
        )

        action = policy_action(policy, obs)
        vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = action

        vx_b += (vx_cmd - vx_b) * dt / tau_v
        vy_b += (vy_cmd - vy_b) * dt / tau_v
        vz += (vz_cmd - vz) * dt / tau_z
        yaw_rate += (yaw_rate_cmd - yaw_rate) * dt / tau_yaw

        x += (math.cos(yaw) * vx_b - math.sin(yaw) * vy_b) * dt
        y += (math.sin(yaw) * vx_b + math.cos(yaw) * vy_b) * dt
        z = max(0.0, z + vz * dt)
        yaw = wrap_angle(yaw + yaw_rate * dt)

        if step >= last_half_start:
            pos_errors.append(pos_error)
            yaw_errors.append(abs(eyaw))
            action_costs.append(float(np.dot(action, action)))

    mean_pos = float(np.mean(pos_errors))
    mean_yaw = float(np.mean(yaw_errors))
    mean_action = float(np.mean(action_costs))
    final_pos = pos_errors[-1]
    reward = -mean_pos - 0.05 * mean_yaw - 0.001 * mean_action - 0.1 * final_pos
    return reward, mean_pos, final_pos


def evaluate(policy, targets):
    rewards = []
    mean_errors = []
    final_errors = []
    for target in targets:
        reward, mean_pos, final_pos = rollout(policy, target)
        rewards.append(reward)
        mean_errors.append(mean_pos)
        final_errors.append(final_pos)
    return float(np.mean(rewards)), float(np.mean(mean_errors)), float(np.mean(final_errors))


def policy_to_json(policy):
    return {
        key: value.tolist()
        for key, value in policy.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-file", default="targets.csv")
    parser.add_argument("--out", default="rl_neural_policy_weights.json")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--episodes-per-candidate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_targets = load_targets(args.targets_file)
    base_policy = default_policy_np()
    mean = flatten_residual(base_policy)
    std = np.full_like(mean, 0.035)

    best_vector = mean.copy()
    best_policy = unflatten_residual(best_vector, base_policy)
    best_reward, best_mean_error, best_final_error = evaluate(best_policy, all_targets)

    elite_count = max(2, int(args.population * args.elite_frac))
    history = []

    for generation in range(1, args.generations + 1):
        candidates = []
        for _ in range(args.population):
            vector = mean + std * rng.standard_normal(mean.shape)
            policy = unflatten_residual(vector, base_policy)
            target_idx = rng.choice(
                len(all_targets),
                size=min(args.episodes_per_candidate, len(all_targets)),
                replace=False,
            )
            targets = [all_targets[i] for i in target_idx]
            reward, mean_error, final_error = evaluate(policy, targets)
            candidates.append((reward, mean_error, final_error, vector))

        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = candidates[:elite_count]
        elite_vectors = np.array([item[3] for item in elites])

        mean = elite_vectors.mean(axis=0)
        std = elite_vectors.std(axis=0) + 0.01
        std = np.clip(std, 0.005, 0.08)

        gen_best_vector = candidates[0][3]
        gen_best_policy = unflatten_residual(gen_best_vector, base_policy)
        full_reward, full_mean_error, full_final_error = evaluate(gen_best_policy, all_targets)

        if full_reward > best_reward:
            best_reward = full_reward
            best_mean_error = full_mean_error
            best_final_error = full_final_error
            best_vector = gen_best_vector.copy()
            best_policy = gen_best_policy

        row = {
            "generation": generation,
            "best_reward": best_reward,
            "best_mean_error": best_mean_error,
            "best_final_error": best_final_error,
            "generation_candidate_reward": full_reward,
            "generation_candidate_mean_error": full_mean_error,
        }
        history.append(row)
        print(
            f"gen={generation:02d} best_mean={best_mean_error:.5f} "
            f"candidate_mean={full_mean_error:.5f} best_reward={best_reward:.5f}"
        )

    out = {
        "algorithm": "cross_entropy_policy_search",
        "description": "Reward-trained residual neural actor for controller_rl_neural.py",
        "seed": args.seed,
        "targets_file": args.targets_file,
        "best_reward": best_reward,
        "best_mean_error_surrogate": best_mean_error,
        "best_final_error_surrogate": best_final_error,
        "policy": policy_to_json(best_policy),
        "history": history,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved trained policy to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
