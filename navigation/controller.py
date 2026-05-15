"""Simple 2D potential field controller for an ego robot.

The ego is modeled as a single integrator: position updates by velocity * dt.
At each step, the velocity is the sum of an attractive force toward the goal
and repulsive forces away from predicted obstacle positions over a short
future horizon. Each future step k contributes gamma^k * phi(ego, o_hat_{t+k}),
where phi is the standard 1/dist repulsive kernel.
"""
import numpy as np


def compute_velocity(ego_pos, goal, obstacle_predictions,
                     k_att, k_rep, influence_radius, max_speed, gamma):
    """Return a velocity vector (2,) for the current step.

    ego_pos:              (2,) current ego position in global coords
    goal:                 (2,) goal position in global coords
    obstacle_predictions: (P, H, 2) array — for each of P obstacles, H predicted
                          future positions o_hat_{t+1..t+H}. Pass shape (0, H, 2)
                          when there are no obstacles.
    k_att:                attractive gain
    k_rep:                repulsive gain
    influence_radius:     obstacles only push the ego when closer than this
    max_speed:            velocity magnitude is clipped to this
    gamma:                discount factor in (0, 1] applied per future step
    """
    # Attractive: unit vector toward the goal, scaled by gain.
    to_goal = goal - ego_pos
    dist_goal = np.linalg.norm(to_goal)
    if dist_goal < 1e-6:
        attractive = np.zeros(2)
    else:
        attractive = k_att * to_goal / dist_goal

    # Repulsive: discounted sum over the prediction horizon, fully vectorized.
    repulsive = np.zeros(2)
    if obstacle_predictions.size > 0:
        # diff[p, k] = ego_pos - o_hat_{t+k}   shape (P, H, 2)
        diff = ego_pos - obstacle_predictions
        dist = np.linalg.norm(diff, axis=-1)              # (P, H)

        # Only obstacles strictly inside the influence radius push the ego.
        # The lower bound guards divide-by-zero when an obstacle lands on top
        # of the ego.
        active = (dist > 1e-6) & (dist < influence_radius)

        # Replace inactive distances with 1.0 so the math below is finite;
        # the contribution is zeroed out via the active mask afterwards.
        safe_dist = np.where(active, dist, 1.0)

        # Standard potential field magnitude: blows up as dist -> 0, zero at
        # the boundary.
        magnitude = k_rep * (1.0 / safe_dist - 1.0 / influence_radius) / (safe_dist * safe_dist)

        # Unit vector from each predicted obstacle position toward the ego.
        unit = diff / safe_dist[..., None]                # (P, H, 2)

        # Discount weights gamma^k for k = 1..H.
        horizon = obstacle_predictions.shape[1]
        weights = gamma ** np.arange(1, horizon + 1, dtype=np.float64)  # (H,)

        # Combine: mask out inactive entries, apply gamma^k, sum over both
        # obstacles and horizon steps.
        force = magnitude[..., None] * unit               # (P, H, 2)
        force = force * active[..., None]                 # zero where inactive
        force = force * weights[None, :, None]            # gamma^k weighting
        repulsive = force.sum(axis=(0, 1))

    velocity = attractive + repulsive

    # Clip to max speed so the ego does not run off when forces are huge.
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity = velocity / speed * max_speed
    return velocity


def at_goal(ego_pos, goal, tol):
    """True when the ego is within tol meters of the goal."""
    return np.linalg.norm(goal - ego_pos) < tol
