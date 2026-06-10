# Drone PID and RL Controllers

This repository contains two controller approaches for the Tello drone simulator:

1. A PID-style lab controller.
2. A neural reinforcement-learning controller experiment.

Both controllers use the same simulator-facing function signature:

```python
controller(state, target_pos, dt, wind_enabled=False)
```

and return:

```python
(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
```

## Files

```text
controllers/controller_pidlab.py
```

PID-style controller with CSV logging.

```text
controllers/controller_rl_neural.py
```

Neural residual policy controller. It loads trained weights from:

```text
training/rl_neural_policy_weights.json
```

```text
training/train_rl_neural_policy.py
```

Reward-based policy-search training script for the neural controller.

```text
reports/RL_Controller_Report.md
```

Short report explaining the RL controller design, training process, and results.

```text
results/rl_neural_assignment_summary.csv
```

PyBullet evaluation summary for the neural controller on the assignment targets.

## RL Training Summary

The neural policy uses:

```text
12 input features -> 12 tanh hidden neurons -> 4 output actions
```

Training used Cross-Entropy Method policy search:

```text
10 generations
40 candidate policies per generation
8 target episodes per candidate
approximately 3200 training episodes
```

The trained neural controller achieved:

```text
Overall mean position error: 0.00850 m
Worst target mean error:     0.01226 m
Targets over 0.01 m:         5 / 24
```

The tuned classical cascade controller remained more accurate, but this repository shows a working neural RL-style alternative and its training pipeline.

## Plots

### PID Tracking

![PID tracking overview](plots/pid_tracking_overview.png)

### Velocity Tuning Before/After

![Velocity tuning position error](plots/velocity_tuning_position_error.png)

![Velocity tuning X response](plots/velocity_tuning_x_stacked.png)

### Simulation vs Experiment

![Simulation vs experiment speed 15](plots/simulation_vs_experiment_speed15.png)

### RL Neural Controller

![RL neural mean error by target](plots/rl_neural_mean_error_by_target.png)
