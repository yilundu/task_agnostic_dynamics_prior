# import matplotlib
# matplotlib.use('TKAgg')

import pybullet as p
import numpy as np
from gym import spaces
from .utils import construct_3d_space, get_3d_image, spawn_rand_goal_3d, step_3d_simulation, respawn_rand_goal, reset_3d_velocity, save_bullet_state, restore_bullet_state
from scipy.misc import imsave
import time
from scipy.misc import imsave
import pybullet_data


class PhysEnv3D(object):

    def __init__(self, order=1, frame_stack=1, pm_oracle=False, pred_steps=0, balance_frames=False):
        self.frame_stack = frame_stack
        self.pm_oracle = pm_oracle
        self.pred_steps = pred_steps
        self.balance_frames = balance_frames

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(fixedTimeStep=1/15., numSolverIterations=1, physicsClientId=self.client)
        self.vs = [(-1, 0), (1, 0 ), (0, -1), (0, 1)]
        self.spec = "trash"


        if not pm_oracle:
            pred_steps = 0

        if not balance_frames:
            self.observation_space = spaces.Box(0, 255, (84, 84, 3*(frame_stack+pred_steps)), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(0, 255, (84, 84, 3*(frame_stack)), dtype=np.uint8)

        self.action_space = spaces.Discrete(4)

    def step(self, actions):

        actions = np.array(actions).flatten()

        reward = 0
        total_length = 0

        info = {}

        obs = [get_3d_image(physicsClientId=self.client)]

        self.iter += 1

        for act in actions:
            pos, ori = p.getBasePositionAndOrientation(self.player_body, physicsClientId=self.client)
            pos = list(pos)
            pos[0] = pos[0] + self.vs[act][0]
            pos[1] = pos[1] + self.vs[act][1]
            pos[2] = pos[2]

            p.resetBasePositionAndOrientation(self.player_body, pos, ori, physicsClientId=self.client)

        goal_pos, _ = p.getBasePositionAndOrientation(self.goal_body, physicsClientId=self.client)

        dist = sum(abs(gp - p) for (gp, p) in zip(goal_pos, pos))

        if dist <= 1.5:
            reward = 1
            respawn_rand_goal(self.goal_body, physicsClientId=self.client)
        else:
            reward = 0

        done = step_3d_simulation(self.player_body, physicsClientId=self.client)

        if self.out_bounds(pos):
            done = True

        ob = self.construct_obs(obs)
        self.total_reward += reward
        self.total_length += total_length

        reset_3d_velocity(self.bodies, physicsClientId=self.client)

        if self.iter > 1000:
            done = True

        if done:
            episode = {}
            episode['r'] = self.total_reward
            episode['l'] = self.total_length
            info['episode'] = episode

        return ob, reward, done, info

    def out_bounds(self, pos):
        return (pos[0] < -0.5) or (pos[0] > 10.5) or (pos[1] < -0.5) or (pos[1] > 10.5)

    def seed(self, seed):
        np.random.seed(seed)
        self.seed = seed

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        bodies = construct_3d_space(min_obj=30, physicsClientId=self.client)
        self.bodies = bodies
        p.setGravity(0,0,-10, physicsClientId=self.client)

        self.iter = 0

        # Construct the player sprite
        player_vid =  p.createVisualShape(p.GEOM_SPHERE, radius = 0.5, rgbaColor=(0.3, 0.3, 0.3, 1), physicsClientId=self.client)
        player_cid =  p.createCollisionShape(p.GEOM_SPHERE, radius = 0.5, physicsClientId=self.client)
        player_body = p.createMultiBody(1, baseVisualShapeIndex = player_vid, baseCollisionShapeIndex = player_cid, basePosition = [5, 5, 0.5], baseOrientation=[0, 0, 0, 1], physicsClientId=self.client)

        goal_body = spawn_rand_goal_3d(first=True, physicsClientId=self.client)

        self.player_body = player_body
        self.goal_body = goal_body

        # g = space.add_collision_handler(collision_types['player'], collision_types['object'])
        # g.begin = remove_player

        # self.space = space
        # self.screen = screen
        # self.draw_options = draw_options
        # self.player_body = body

        self.total_reward = 0
        self.total_length = 0

        self.frame_buffer = []

        for i in range(self.frame_stack):
            step_3d_simulation(self.player_body, physicsClientId=self.client)
            self.frame_buffer.append(get_3d_image(physicsClientId=self.client))

        obs = self.construct_obs()
        return obs

    def render(self):
        im = get_3d_image(physicsClientId=self.client)
        import matplotlib.pyplot as plt
        plt.imshow(im)
        plt.show()

    def construct_obs(self, new_obs=None):
        # Adds a frame into the frame buffer and returns the concatenated observation
        if new_obs is None:
            new_obs = [get_3d_image(physicsClientId=self.client)]

        # print(p.getBasePositionAndOrientation(1))
        # print(p.getBasePositionAndOrientation(2))
        self.frame_buffer.extend(new_obs)
        self.frame_buffer = self.frame_buffer[-self.frame_stack:]
        obs = self.frame_buffer

        if self.pred_steps > 0 and self.pm_oracle:
            snapshot = save_bullet_state(physicsClientId=self.client)
            future_obs = []

            for i in range(self.pred_steps):
                step_3d_simulation(self.player_body, physicsClientId=self.client)
                future_obs.append(get_3d_image(physicsClientId=self.client))

            restore_bullet_state(snapshot, physicsClientId=self.client)
            obs.extend(future_obs)

        # self.player_body.reward = 0

        if self.balance_frames:
            obs = obs[-self.frame_stack:]

        return np.concatenate(obs, axis=2)

    def close(self):
        pass


def fetch_action():
    return int(input("Enter your move (0, 1, 2, 3) : "))


if __name__ == "__main__":
    import imageio
    env = PhysEnv3D()
    ims = []
    im = env.reset()
    ims.append(im)

    done = False
    old_obs = 0
    obs = 0
    for i in range(100):
        obs, reward, done, _ = env.step(2)
        imsave("test.png", obs)
        ims.append(obs)

    imageio.mimsave("test.gif", ims)
