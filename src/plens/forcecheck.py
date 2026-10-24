"""Is wrist force/torque available and plausible in robosuite's Square task? (docs/tasks.md, ch. 13)

Needs only robosuite (plens-robosuite env: robosuite 1.5.2 with mujoco==3.3.0; newer
MuJoCo fails a joint-type assertion in robosuite).
"""

from __future__ import annotations

import numpy as np


def press_test(steps_down: int = 45, steps_total: int = 60, env_name: str = "NutAssemblySquare") -> np.ndarray:
    """Drive a Panda's gripper straight down onto the table; return |wrist force| (N) per step.

    Expect a few N while the gripper hangs free (its own weight), then a jump to hundreds of N
    once it presses on the table: MuJoCo contact is stiff, so sim force is spiky.
    """
    import robosuite as suite

    env = suite.make(env_name, robots="Panda", has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20, horizon=steps_total + 10)
    env.reset()
    robot = env.robots[0]
    forces = []
    for t in range(steps_total):
        a = np.zeros(env.action_dim)
        a[2] = -1.0 if t < steps_down else 0.0  # z velocity command: down, then hold
        env.step(a)
        f = robot.ee_force["right"] if isinstance(robot.ee_force, dict) else robot.ee_force
        forces.append(float(np.linalg.norm(f)))
    env.close()
    return np.array(forces)
