# Reinforcement Learning Controller Report

## Overview

This report describes the reinforcement-learning version of the drone feedback controller created for the Tello simulator. The original submission-ready controller is `controller_lab.py`, which uses a tuned cascaded control structure. The RL work was developed separately so that the proven submission controller was not changed.

The RL version was implemented as a neural residual policy. It keeps the same simulator interface:

```python
controller(state, target_pos, dt, wind_enabled=False)
```

and returns:

```python
(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
```

This means the RL controller can be swapped into the simulator without changing the simulator code.

## Baseline Controller

The baseline controller in `controller_lab.py` uses a cascaded structure:

```text
position error -> desired velocity -> corrected velocity command
```

The outer position loop converts position error into velocity setpoints. During the approach phase it is mainly proportional. A small integral term is enabled only after the drone has settled near the target, which prevents integral windup during large movements.

The inner velocity loop compares desired velocity with estimated velocity and adds damping. This reduces overshoot without requiring a derivative term in the outer position loop.

The controller outputs body-frame velocity commands and a yaw-rate command. These are passed into the simulator's built-in Tello controller, which handles lower-level velocity, attitude, rate, and motor control.

## RL Controller Design

The neural RL controller is implemented in:

```text
controller_rl_neural.py
```

The training script is:

```text
train_rl_neural_policy.py
```

The saved trained policy weights are:

```text
rl_neural_policy_weights.json
```

The neural controller uses a small actor network:

```text
12 input features -> 12 tanh hidden neurons -> 4 output actions
```

The input features include:

```text
body-frame x error
body-frame y error
z error
yaw error
estimated body-frame x velocity
estimated body-frame y velocity
estimated z velocity
yaw error rate
integral/memory terms
```

The four output actions are:

```text
vx_cmd
vy_cmd
vz_cmd
yaw_rate_cmd
```

For safety and stability, the neural network was used as a residual actor. A stable base policy provides reasonable velocity commands, and the neural network learns an additional correction. This is safer than training a fully unconstrained neural controller from scratch because the drone still has a stable fallback behaviour.

## Training Method

The policy was trained using reward-based policy search with the Cross-Entropy Method. This is an evolutionary reinforcement-learning approach:

1. Generate many candidate neural policies.
2. Run simulated episodes for each candidate.
3. Score each candidate using a reward function.
4. Keep the best candidates.
5. Update the policy distribution toward the best candidates.
6. Repeat for several generations.

Training configuration:

```text
Generations: 10
Population per generation: 40 policies
Episodes per candidate: 8 targets
Approximate training episodes: 10 x 40 x 8 = 3200
```

The reward penalised:

```text
mean position error
yaw error
large control effort
final position error
```

Training used a lightweight first-order drone model so that thousands of policy evaluations could be run quickly. The final trained policy was then tested in the full PyBullet simulator.

## Results

The trained neural RL controller was tested on the 24 assignment targets in PyBullet.

Results for `controller_rl_neural.py`:

```text
Overall mean position error: 0.00850 m
Worst target mean error:     0.01226 m
Targets over 0.01 m:         5 / 24
```

The detailed result table was saved in:

```text
rl_neural_assignment_summary.csv
```

For comparison, the tuned cascaded controller in `controller_lab.py` performed better overall. The RL controller worked and achieved an overall mean error below 0.01 m, but it did not outperform the hand-tuned cascaded controller.

## Discussion

The RL controller demonstrates that the control problem can be represented as a neural policy. It successfully maps observations to velocity commands and can run inside the same simulator API as the normal controller.

However, the cascaded controller remains better for submission because it was carefully tuned for this simulator and has predictable structure. The neural RL controller would likely need longer training, more accurate training dynamics, and more policy search iterations to outperform the classical controller.

The main advantage of the RL approach is flexibility. A neural policy can learn corrections that are difficult to hand-design, especially if trained directly in the full simulator or on real experimental data. The disadvantage is that training is more expensive and the resulting behaviour is less interpretable than the cascaded controller.

## Conclusion

A working neural RL-style controller was created and tested. It uses a trained residual neural policy with saved weights and the same interface as the original controller. The controller achieved an overall mean position error below 0.01 m on the 24 assignment targets, showing that the approach works. However, the original `controller_lab.py` cascaded controller remains the recommended submission because it is more accurate, simpler, and more reliable.

