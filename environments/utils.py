import gym
import numpy as np


def find_farthest(reference, positions):
    """Find the position that is farthest away from the reference."""
    distances = [np.linalg.norm(reference - position) for position in positions]
    index = np.argmax(distances)
    
    return positions[index]


def poisson2d(x, y, tau=15):
    """Two-dimensional Poisson window function."""
    r = np.sqrt(np.square(x) + np.square(y))
    return np.exp(-np.abs(r) / tau)


def subcell_of(cell, start, end):
    """Extract a subcell `[start, end]` from an Ocelot cell."""
    subcell = []
    is_in_subcell = False
    for el in cell:
        if el.id == start: is_in_subcell = True
        if is_in_subcell: subcell.append(el)
        if el.id == end: break
    
    return subcell


def unwrap(env):
    """Unwrap (i.e. remove all wrappers) a wrapped environment to the bare environment itself."""
    return env if not isinstance(env, gym.Wrapper) else unwrap(env.env)
