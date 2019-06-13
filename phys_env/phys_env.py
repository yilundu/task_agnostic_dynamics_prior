import numpy as np
import pymunk as pm
import pymunk.pygame_util
from .utils import construct_default_space, COLORS_TUPLE, step_simulation, collision_types, save_state, restore_state, spawn_rand_goal, distance
import matplotlib.pyplot as plt
import pygame
from easydict import EasyDict
import numpy as np
from gym import spaces
import random


class PhysEnv(object):

    def __init__(self, order=1, frame_stack=1, pm_oracle=False, pred_steps=0, balance_frames=False):
        self.frame_stack = frame_stack
        self.pm_oracle = pm_oracle
        self.pred_steps = pred_steps
        self.balance_frames = balance_frames
        self.reset()
        self.vs = [(-500, 0), (500, 0 ), (0, -500), (0, 500)]
        self.spec = "trash"

        if not pm_oracle:
            pred_steps = 0

        if not balance_frames:
            self.observation_space = spaces.Box(0, 255, (84, 84, 3*(frame_stack+pred_steps)), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(0, 255, (84, 84, 3*(frame_stack)), dtype=np.uint8)

        self.order = order
        self.action_space = spaces.Discrete(4)

    def step(self, actions):

        actions = np.array(actions).flatten()

        reward = 0
        total_length = 0

        info = {}

        obs = []

        for act in actions:
            dv = self.vs[act]
            v = self.player_body.velocity
            nv = (dv[0], dv[1])
            self.player_body.velocity = nv

            self.space.step(0.01)
            obs.append(step_simulation(self.screen, self.space, self.draw_options))
            total_length += 1

            if self.player_body.done or self.out_bounds(self.player_body.position):
                reward = -1
                self.total_reward += reward
                self.total_length += total_length
                done = True
                episode = {}
                episode['r'] = self.total_reward
                episode['l'] = self.total_length
                info['episode'] = episode
                ob = self.construct_obs(obs)
                return ob, reward, done, info
            else:
                done = False

        if distance(self.player_body, self.target_body) < 10:
            reward = 1
            self.target_body.position = 84 * random.uniform(0.1, 0.9), 84 * random.uniform(0.1, 0.9)

        ob = self.construct_obs(obs)
        self.total_reward += reward
        self.total_length += total_length

        return ob, reward, done, info

    def out_bounds(self, pos):
        return (pos[0] < 0) or (pos[0] > 84) or (pos[1] < 0) or (pos[1] > 84)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        screen, space, draw_options = construct_default_space(min_obj=7)

        # Construct the player sprite
        body = pm.Body(1, 1)
        body.velocity = (0, 0)
        body.position = (6, 6)
        body.done = False
        body.reward = 0

        shape = pm.Circle(body, 4)
        shape.color = (0, 0, 255, 255)
        shape.elasticity = 0.95
        shape.friction = 0.9
        shape.collision_type = collision_types["player"]

        space.add(body, shape)

        t_body = spawn_rand_goal(space, 84)

        self.target_body = t_body

        def remove_player(arbiter, space, data):
            player_shape = arbiter.shapes[0]
            player_shape.body.done = True
            # space.remove(player_shape, player_shape.body)
            return True

        dim = 84

        g = space.add_collision_handler(collision_types['player'], collision_types['object'])
        g.begin = remove_player

        self.space = space
        self.screen = screen
        self.draw_options = draw_options
        self.player_body = body

        self.total_reward = 0
        self.total_length = 0

        self.frame_buffer = list(np.zeros((self.frame_stack, 84, 84, 3), dtype=np.uint8))

        obs = self.construct_obs()
        return obs

    def render(self):
        im = step_simulation(self.screen, self.space, self.draw_options)
        plt.imshow(im)
        plt.show()

    def construct_obs(self, new_obs=None):
        # Adds a frame into the frame buffer and returns the concatenated observation
        if new_obs is None:
            new_obs = [step_simulation(self.screen, self.space, self.draw_options)]

        self.frame_buffer.extend(new_obs)
        self.frame_buffer = self.frame_buffer[-self.frame_stack:]
        obs = self.frame_buffer

        if self.pred_steps > 0 and self.pm_oracle:
            snapshot = save_state(self.space)
            future_obs = []

            for i in range(self.pred_steps):
                self.space.step(0.01)
                future_obs.append(step_simulation(self.screen, self.space, self.draw_options))

            restore_state(snapshot)
            obs.extend(future_obs)

        self.player_body.reward = 0

        if self.balance_frames:
            obs = obs[-self.frame_stack:]

        return np.concatenate(obs, axis=2)

    def close(self):
        pass


def fetch_action():
    return int(input("Enter your move (0, 1, 2, 3) : "))


if __name__ == "__main__":
    env = PhysEnv()

    done = False
    while not done:
        # get action
        action = fetch_action()
        _, reward, done, _ = env.step(action)
        print(reward)
        env.render()
