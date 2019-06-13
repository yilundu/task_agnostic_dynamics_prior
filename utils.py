import tensorflow as tf
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, EpisodicLifeEnv, FireResetEnv, \
                                     ScaledFloatFrame, ClipRewardEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import set_global_seeds
from baselines.bench import Monitor
from baselines import logger
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
import random
import os
from torchvision.transforms import functional
from PIL import Image

# from dm_control import suite
# from dm_control.suite.wrappers import pixels
from gym import spaces
from collections import OrderedDict

# FLAGS = flags.FLAGS

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, size=84, keep_obs=False):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = size
        self.height = size

        if keep_obs:
            self.observation_space = env.observation_space
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):

        if type(frame) == OrderedDict:
            frame['pixels'] = cv2.resize(frame['pixels'], (self.width, self.height), interpolation=cv2.INTER_AREA)

        else:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

class AugmentColor(gym.Wrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.Wrapper.__init__(self, env)
        self._reset_random()


    def reset(self):
        self._reset_random()
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = Image.fromarray(ob)
        ob = functional.adjust_brightness(ob, self.brightness)
        ob = functional.adjust_contrast(ob, self.contrast)
        ob = functional.adjust_saturation(ob, self.saturation)
        ob = functional.adjust_hue(ob, self.hue)
        ob_new = np.array(ob)
        # ob_new = np.clip(ob.astype(np.float32) + np.random.uniform(-10, 10, ob.shape), 0, 255).astype(np.uint8)
        return ob_new, reward, done, info

    def _reset_random(self):
        self.brightness = np.random.uniform(0.7, 1.3)
        self.contrast = np.random.uniform(0.7, 1.3)
        self.saturation = np.random.uniform(0.7, 1.3)
        self.hue = np.random.uniform(0.0, 0.3)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, keep_obs=False):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        rl_common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape

        if keep_obs:
            self.observation_space = env.observation_space
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()

        if type(ob) == OrderedDict:
            ob_frame = ob['pixels']
        else:
            ob_frame = ob
        for _ in range(self.k):
            self.frames.append(ob_frame)
        return self._get_ob(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if type(ob) == OrderedDict:
            self.frames.append(ob['pixels'])
        else:
            self.frames.append(ob)
        return self._get_ob(ob), reward, done, info

    def _get_ob(self, ob):
        assert len(self.frames) == self.k

        if type(ob) == OrderedDict:
            ob['pixels'] = np.concatenate(self.frames, axis=2)
            return ob
        else:
            return np.concatenate(self.frames, axis=2)


class RandomRepeat(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        while (not done) and (random.uniform(0, 1) < 0.5):
            ob, temp_reward, done, info = self.env.step(action)
            reward += temp_reward

        return ob, reward, done, info



class EpsRandom(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n = env.action_space.n

    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, action):
        if (random.uniform(0, 1) < 0.1):
            action = random.randint(0, self.n-1)

        ob, reward, done, info = self.env.step(action)

        return ob, reward, done, info


class MakeGym():
    def __init__(self, env):
        # self.n = env.action_space.n
        act_spec = env.action_spec()
        obs_spec = env.observation_spec()

        total_dim = 0
        for name in list(obs_spec):
            if name != "pixels":
                if len(obs_spec[name].shape) == 0:
                    total_dim += 1
                else:
                    total_dim += obs_spec[name].shape[0]

        self.observation_space = spaces.Box(low=-100., high=100., shape=(total_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=act_spec.minimum, high=act_spec.maximum, dtype=np.float32)
        self.reward_range = (0, 1)
        self.metadata = {}
        self.env = env
        self.spec = "shit"

        # Max length of 1000
        self.counter = 0

    def reset(self):
        timestep, reward, discount, ob = self.env.reset()
        self.counter = 0
        return self._construct_ob(ob)

    def _construct_ob(self, ob):
        state = []
        for key in ob.keys():
            if key != 'pixels':
                if type(ob[key]) == np.float64:
                    state.append([ob[key]])
                else:
                    state.append(ob[key])

        ob_flat = np.concatenate(state, axis=0)
        ob_dict = {'pixels': ob['pixels'], 'flat': ob_flat}

        return OrderedDict(ob_dict)

    def step(self, action):
        timestep, reward, discount, ob = self.env.step(action)

        if reward is None:
            reward = 0.0

        done = False

        if self.counter == 1000:
            self.counter = 0
            done = True

        self.counter += 1
        info = {}

        return self._construct_ob(ob), reward, done, info


class RandomFix(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n = env.action_space.n
        self.counter = 0

    def reset(self):
        ob = self.env.reset()
        self.counter = 0
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        if self.counter % 5 == 0:
            for i in range(2):
                if done:
                    break
                action = random.randint(0, self.n-1)
                ob, reward_tmp, done, info = self.env.step(action)
                reward = reward + reward_tmp

        return ob, reward, done, info


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = tf.get_variable(saved_var_name)
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
            except:
                print("Skipping variable {}".format(var_name))
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def make_atari_env_custom(env_id, num_env, seed, frame_stack, wrapper_kwargs={}, start_index=0, random_action=False, eps_random=False, augment=False, clip_rewards=True, episode_life=True, random_fix=False, size=84):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}

    def wrap_deepmind_custom(env, episode_life=True, clip_rewards=True, frame_stack=frame_stack, scale=False):
        if episode_life:
            env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, size=size)
        if augment:
            env = AugmentColor(env)
        if scale:
            env = ScaledFloatFrame(env)
        if clip_rewards:
            env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, frame_stack)
        return env

    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank if seed is not None else None)
            if random_action:
                env = RandomRepeat(env)
            if eps_random:
                env = EpsRandom(env)
            if random_fix:
                env = RandomFix(env)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_deepmind_custom(env, episode_life=episode_life, clip_rewards=clip_rewards, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_dm_control(domain_name, task_name, num_env, seed, frame_stack, vis_reward=False, wrapper_kwargs={}, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}

    def wrap_env(seed):

        env = suite.load(domain_name, task_name, task_kwargs={'random':seed}, visualize_reward=vis_reward)
        env = pixels.Wrapper(env, pixels_only=False)
        env = MakeGym(env)
        env = WarpFrame(env, keep_obs=True)
        env = FrameStack(env, frame_stack, keep_obs=True)
        return env

    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = wrap_env(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def wrap_pad(input, size):
    M1 = tf.concat([input[:,:, -size:,:], input, input[:,:, 0:size,:]], 2)
    M1 = tf.concat([M1[:,:, :,-size:], M1, M1[:,:, :,0:size]], 3)
    return M1


def spatial_mem(inp, state, FLAGS, reuse=False, scope="", bypass_res=False, action=None):
    """Computes representation on inp through spatial mem given inp of size nchw and state of nchw"""

    state = state[0]
    state_shape = state.get_shape()
    state_channel = state.get_shape()[1]
    input_channel = inp.get_shape()[1]
    merge = tf.concat([inp, state], axis=1)
    merge_dim = state_channel + input_channel

    if action is not None:
        state = tf.concat([action, state], axis=1)

    with tf.variable_scope("spatial_mem"+scope, reuse=reuse):
        if not bypass_res:
            state_res = tf.layers.conv2d(inputs=merge, filters=state_channel, kernel_size=(5, 5), strides=(1, 1),
                                     padding='same', name='mem1', data_format='channels_first', activation=tf.nn.elu)
            state_concat = tf.concat([state, state_res], axis=1)
            state = tf.layers.conv2d(inputs=state_concat, filters=state_channel, kernel_size=(5, 5), strides=(1,1),
                                     padding='same', name='state_merge', data_format='channels_first', activation=tf.nn.elu)

        # state = wrap_pad(state, 2)
        state_step = state_next = tf.layers.conv2d(inputs=state, filters=state_channel, kernel_size=(5, 5), strides=(1, 1),
                                 padding='same', name='dym1', data_format='channels_first', activation=tf.nn.elu)

        outputs = []

        for i in range(1):

            merge_step_1 = tf.concat([state_step, inp], axis=1)
            # merge_step_1 = wrap_pad(merge_step_1, 1)
            output = tf.layers.conv2d(inputs=merge_step_1, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same', name='output', data_format='channels_first', reuse=reuse, activation=tf.nn.elu)
            outputs.append(output)

            # if i != FLAGS.pred_steps -1:
            #     state_step = wrap_pad(state_step, 2)
            #     state_step = tf.layers.conv2d(inputs=state_step, filters=state_channel, kernel_size=(5, 5), strides=(1, 1),
            #                                   padding='same', name='dym1', data_format='channels_first', reuse=True, activation=tf.nn.elu)
            #     reuse = True

        output = tf.stack(outputs, axis=1)


    return [state_next], output, state_step


def convlstm(inp, state, FLAGS, reuse=False, scope="", bypass_res=False):
    """Computes representation on inp through spatial mem given inp of size nchw and state of nchw"""

    state = state[0]
    cell, hidden = tf.split(state, 2, axis=1)
    input_channel = cell.get_shape()[1]
    output_channel = inp.get_shape()[1]

    with tf.variable_scope("spatial_lstm"+scope, reuse=reuse):
        input_x = tf.layers.conv2d(inputs=inp, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='ig', data_format='channels_first', reuse=reuse, use_bias=False)
        input_h = tf.layers.conv2d(inputs=hidden, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='hg', data_format='channels_first', reuse=reuse, use_bias=False)
        input_c = tf.layers.conv2d(inputs=cell, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='cg', data_format='channels_first', reuse=reuse)

        inp_gate = tf.nn.sigmoid(input_x + input_h + input_c)

        forget_x = tf.layers.conv2d(inputs=inp, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='if', data_format='channels_first', reuse=reuse, use_bias=False)
        forget_h = tf.layers.conv2d(inputs=hidden, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='hf', data_format='channels_first', reuse=reuse, use_bias=False)
        forget_c = tf.layers.conv2d(inputs=cell, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='cf', data_format='channels_first', reuse=reuse)

        forget_gate = tf.nn.sigmoid(forget_x + forget_h + forget_c)


        input_act_x = tf.layers.conv2d(inputs=inp, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='ii', data_format='channels_first', reuse=reuse, use_bias=False)
        input_act_h = tf.layers.conv2d(inputs=hidden, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='hi', data_format='channels_first', reuse=reuse)

        inp = input_act_x + input_act_h

        output_x = tf.layers.conv2d(inputs=inp, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='io', data_format='channels_first', reuse=reuse, use_bias=False)
        output_h = tf.layers.conv2d(inputs=hidden, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='ho', data_format='channels_first', reuse=reuse, use_bias=False)
        output_c = tf.layers.conv2d(inputs=cell, filters=input_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='co', data_format='channels_first', reuse=reuse)

        output_gate = tf.nn.sigmoid(output_x + output_h + output_c)

        cell_new = forget_gate * cell + inp_gate * tf.nn.tanh(inp)

        hidden_new = tf.nn.tanh(output_gate * cell_new)
        output = tf.layers.conv2d(inputs=hidden_new, filters=output_channel, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='creshape', data_format='channels_first', reuse=reuse)
        state_next = tf.concat([cell_new, hidden_new], axis=1)

    return [state_next], output, None


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, name):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format, name=name)

def batch_norm(inputs, training, data_format, name):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True, name=name)


def residual_block(inputs, filters, training, data_format, name='', reuse=False, use_batch=False):

    with tf.variable_scope(name, reuse=reuse):
        shortcut = inputs

        if use_batch:
            inputs = batch_norm(inputs, training, data_format, name='bn1')

        inputs = tf.nn.leaky_relu(inputs)

        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format, name='conv1')

        if use_batch:
            inputs = batch_norm(inputs, training, data_format, name='bn2')

        inputs = tf.nn.leaky_relu(inputs)
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format, name='conv2')

    return inputs + shortcut


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.stop_gradient(tf.reshape(lab_pixels, tf.shape(srgb)))

def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return tf.stop_gradient(tf.stack([L_chan / 50 - 1, a_chan / 110, b_chan / 110], axis=3))


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        ims = []
        for i in idxes:
            ims.append(self._storage[i])
        return np.array(ims)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


