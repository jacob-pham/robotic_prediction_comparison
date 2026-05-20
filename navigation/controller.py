"""Potential field controller for a 2D ego robot.

The ego moves as a single integrator (pos += vel * dt). Each step the velocity
is an attractive pull to the goal plus a discounted sum of repulsive pushes
from predicted obstacle positions over a short horizon.
"""
import numpy as np


def compute_velocity(ego_pos, goal, obstacle_predictions,
                     k_att, k_rep, influence_radius, max_speed, gamma):
    """Velocity for the current step.

    input:
        ego_pos: (2,) current ego position
        goal: (2,) goal position
        obstacle_predictions: (P, H, 2) future positions per obstacle, or (0, H, 2)
        k_att, k_rep: attractive and repulsive gains
        influence_radius: obstacles inside this radius push the ego
        max_speed: velocity magnitude is clipped to this
        gamma: per-step discount in (0, 1]
    output:
        (2,) velocity vector, clipped to max_speed
    """
    # pull toward the goal
    to_goal = goal - ego_pos
    dist_goal = np.linalg.norm(to_goal)
    if dist_goal < 1e-6:
        attractive = np.zeros(2) # already at the goal, no attractive force
    else:
        attractive = k_att * to_goal / dist_goal # unit vector toward the goal, scaled by gain

    # repulsive push away from each predicted obstacle position
    # we loop over every obstacle and every step of its predicted future, add
    # up a small push for each one, and discount steps further in the future
    repulsive = np.zeros(2)
    if obstacle_predictions.size > 0:
        num_obstacles = obstacle_predictions.shape[0]
        horizon = obstacle_predictions.shape[1]
        for i in range(num_obstacles):
            for h in range(horizon):
                obstacle_pos = obstacle_predictions[i, h]
                # vector pointing from the obstacle toward the ego
                diff = ego_pos - obstacle_pos
                dist = np.linalg.norm(diff)

                # skip obstacles that are basically on top of us (avoid divide
                # by zero) or too far away to matter
                if dist <= 1e-6 or dist >= influence_radius:
                    continue

                # standard potential field repulsion magnitude
                magnitude = k_rep * (1.0 / dist - 1.0 / influence_radius) / (dist * dist)
                direction = diff / dist  # unit vector pointing away from obstacle
                weight = gamma ** (h + 1)  # discount steps further in the future
                repulsive = repulsive + magnitude * direction * weight

    velocity = attractive + repulsive

    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity = velocity / speed * max_speed # clip to max_speed
    return velocity


def at_goal(ego_pos, goal, tol):
    """Check whether the ego has reached the goal.

    input:
        ego_pos: (2,) current ego position
        goal: (2,) goal position
        tol: distance tolerance (meters)
    output:
        True if ego is within tol of the goal, else False
    """
    return np.linalg.norm(goal - ego_pos) < tol
