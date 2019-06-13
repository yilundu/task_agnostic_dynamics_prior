import pymunk as pm
import pymunk.pygame_util
import pygame
from pygame.color import *
from pygame.locals import *
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import imageio
from multiprocessing.pool import Pool
from utils import construct_default_space, step_simulation, construct_space

NUM_COLORS = 657
COLORS_TUPLE = list(THECOLORS.values())

def generate(dim, num_sim=250):
    screen, space, draw_options = construct_default_space()
    obs = []

    for i in range(num_sim):
        im = step_simulation(screen, space, draw_options)
        obs.append(im)
        space.step(0.01)

    return obs

def generate_simple(dim, num_sim=500):
    """Only generate a simulation with 1 fixed wall location and one random ball"""
    space, screen, draw_options = construct_space(84)
    scene_color = (0, 0, 0)
    pos_low = 0.1
    pos_high = 0.9
    v_scale = 200

    pos = dim * random.uniform(pos_low, pos_high), 10
    velocity = random.uniform(0.1, 1) * v_scale, v_scale * random.uniform(-1, 1)
    body = pm.Body(1, 1)
    body.velocity = velocity
    body.position = pos
    shape = pm.Circle(body, np.random.randint(5, 10))

    shape.elasticity = 0.95
    shape.friction = 0.9
    idx = np.random.randint(0, NUM_COLORS)
    shape.color = tuple(COLORS_TUPLE[idx])

    space.add(body, shape)

    # Fix a horizontal wall at coordinate 70
    start_seg = (10, 70)
    end_seg = (74, 70)
    static_body = space.static_body
    seg = pm.Segment(static_body, start_seg, end_seg, 0.0)
    seg.elasticity = 0.95
    seg.friction = 0.9

    space.add(seg)

    obs = []

    for i in range(num_sim):
        im = step_simulation(screen, space, draw_options)
        obs.append(im)
        space.step(0.01)

    return obs


def generate_traj(args):
    obs = generate(84)
    obs = np.stack(obs)
    obs = obs[::2]

    return obs

def generate_traj_simple(args):
    obs = generate_simple(84)
    obs = np.stack(obs)
    obs = obs[::2]

    return obs

def generate_trajs(traj_num=50000, simple=False):
    args = [None] * traj_num
    pool = Pool()
    if simple:
        total_obs = pool.map(generate_traj_simple, args)
    else:
        total_obs = pool.map(generate_traj, args)

    return np.stack(total_obs)


if __name__ == "__main__":
    pygame.init()
    obs = generate_trajs(simple=False)
    np.savez("test_blank_large.npz", obs)
