"""Simple 2D potential field controller for an ego robot.

The ego is modeled as a single integrator: position updates by velocity * dt.
At each step, the velocity is the sum of an attractive force toward the goal
and repulsive forces away from nearby obstacles.
"""
import numpy as np


def compute_velocity(ego_pos, goal, obstacles,
                     k_att, k_rep, influence_radius, max_speed):
    """Return a velocity vector (2,) for the current step.

    ego_pos:          (2,) current ego position in global coords
    goal:             (2,) goal position in global coords
    obstacles:        list of (2,) arrays — obstacle positions at this step
    k_att:            attractive gain
    k_rep:            repulsive gain
    influence_radius: obstacles only push the ego when they are closer than this
    max_speed:        velocity magnitude is clipped to this
    """
    # Attractive: unit vector toward the goal, scaled by gain.
    to_goal = goal - ego_pos
    dist_goal = np.linalg.norm(to_goal)
    if dist_goal < 1e-6:
        attractive = np.zeros(2)
    else:
        attractive = k_att * to_goal / dist_goal

    # Repulsive: sum contributions from each close obstacle.
    repulsive = np.zeros(2)
    for obs in obstacles:
        diff = ego_pos - obs
        dist = np.linalg.norm(diff)
        if 1e-6 < dist < influence_radius:
            # Standard potential field form: blows up as dist -> 0, zero at the boundary.
            magnitude = k_rep * (1.0 / dist - 1.0 / influence_radius) / (dist * dist)
            repulsive += magnitude * (diff / dist)

    velocity = attractive + repulsive

    # Clip to max speed so the ego does not run off when forces are huge.
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity = velocity / speed * max_speed
    return velocity


def at_goal(ego_pos, goal, tol):
    """True when the ego is within tol meters of the goal."""
    return np.linalg.norm(goal - ego_pos) < tol
