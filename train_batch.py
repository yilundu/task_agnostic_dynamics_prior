import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from utils import make_atari_env_custom, make_dm_control
from baselines.common.distributions import CategoricalPdType, MultiCategoricalPdType
from baselines.common.runners import AbstractEnvRunner
from baselines.common import explained_variance
from baselines.common.tf_util import initialize
from baselines.logger import TensorBoardOutputFormat
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import argparse
from tqdm import tqdm
import time
from collections import deque
from phys_env import phys_env, phys_env_alternate, phys_env_shooter, phys_3d, phys_3d_shooter
from utils import optimistic_restore, AugmentColor
from easydict import EasyDict
from baselines.common.distributions import make_pdtype

import numpy as np
import tensorflow as tf
import os.path as osp
import os

import imageio

from models import PhysNetV2, JunyukRecurrentNet, PhysActionRollout

FLAGS = EasyDict({'balance_frames': True, 'batch': True, 'cliprange': 0.1, 'ent_coeff': 0.01, 'env': 'phys_env', 'exp': 'default', 'finetune_data': '/mnt/nfs/yilundu/phys_prior/test_reduced.npz', 'finetune_epochs': 1, 'finetune_physics': False, 'forward_state_pred': 1, 'gamma': 0.99, 'lam': 0.95, 'log_interval': 1, 'logdir': '/root/results/resource/logs/smoketest', 'lr': 0.00025, 'max_grad_norm': 0.5, 'mixed_finetune': False, 'nenvs': 8, 'nminibatches': 4, 'noptsepochs': 4, 'nsteps': 128, 'num_steps': 32, 'num_timesteps': 10000000.0, 'order': 1, 'pm': False, 'pm_nova': False, 'pm_oracle': False, 'pm_path': '/root/code/policy_physics/policy_physics/physmodel/model_24794', 'pm_rollouts': False, 'pos': False, 'pred_steps': 3, 'random_action': False, 'resume_iter': -1, 'resume_physics_only': False, 'resume_policy_only': False, 'save_iter': 5000, 'seed': 0, 'sonic_joint': False, 'sonic_transfer': False, 'test_phys': True, 'train': True, 'vf_coef': 0.5, 'phys_lr':1e-4, 'model': 'physnet', 'augment': False, 'i2a': False, 'joint_policy': False, 'convlstm': False, 'resume_goal': False, 'resume_forage': False, 'eps_random': False, 'timeskip': 1, 'random_fix': False, 'dm_control': False, 'vis_reward': True, 'blink': False, 'i2a_action': False, 'i2a_action_n': 5})

junyuk_pm_path = '/root/code/policy_physics/policy_physics/physmodel/model_68310'
convlstm_pm_path = '/root/code/policy_physics/policy_physics/physmodel/model_39688'
pm_path_blink = '/root/code/policy_physics/policy_physics/physmodel/model_3630'

assert not (FLAGS.pm and FLAGS.pm_rollouts)


# Use a convolutional network described in Nature paper
class ConvPolicy(object):
    def __init__(self, sess, env, FLAGS, model_fn, reuse=False):
        # Construct the model for training
        print(env.observation_space)

        ob_space = env.observation_space
        self.pdtype = make_pdtype(env.action_space)

        self.states = None

        if FLAGS.pm or FLAGS.pm_rollouts:
            if FLAGS.model == 'physnet':
                self.initial_state = np.zeros((FLAGS.nenvs, 64, 21, 21))
                STATE = tf.placeholder(shape=(None, 64, 21, 21), dtype=tf.float32, name='SpatialMem')
            else:
                self.initial_state = np.zeros((FLAGS.nenvs, 2048))
                STATE = tf.placeholder(shape=(None, 1024*2), dtype=tf.float32, name='SpatialMem')
        else:
            self.initial_state = np.zeros(1)

        STATE = tf.placeholder(shape=(None, 64, 21, 21), dtype=tf.float32, name='SpatialMem')

        X = control_x = tf.placeholder(shape=(None, ) + ob_space.shape, dtype=ob_space.dtype, name='Ob_control')
        X_IM  = tf.placeholder(shape=(None, 84, 84, 12), dtype=ob_space.dtype, name='Ob')
        MASK = tf.placeholder(shape=(None), dtype=tf.float32, name='Mask')
        ACTION = tf.placeholder(shape=(None), dtype=tf.int32, name='action')
        self.ACTION = ACTION

        self.STATE = STATE
        self.MASK = MASK

        if FLAGS.model == 'physnet':
            state_mask = tf.reshape(MASK, (-1, 1, 1, 1))
            state = STATE * (1 - state_mask)
        else:
            state_mask = tf.reshape(MASK, (-1, 1))
            state = STATE * (1 - state_mask)

        if not FLAGS.dm_control:
            X = X_IM

        processed_x = tf.to_float(X_IM)
        processed_x = processed_x / 255.
        processed_x = tf.transpose(processed_x, perm=(0, 3, 1, 2))

        final_state = tf.zeros(1)

        if FLAGS.pm:
            if FLAGS.i2a:
                self.phys_model = model_fn(processed_x, [state], FLAGS, num_steps=FLAGS.pred_steps, reuse=reuse)
                with tf.variable_scope("policy_model", reuse=reuse):
                    final_output = tf.stop_gradient(self.phys_model.final_output)
                    output = tf.reshape(tf.stop_gradient(final_output), (tf.shape(processed_x)[0] * FLAGS.pred_steps, 3, 84, 84))

                    conv1 = tf.layers.conv2d(inputs=tf.stop_gradient(output), filters=32, kernel_size=[3, 3], strides=(8, 8),
                                             padding='same', activation=tf.nn.relu, name='encode_c1', data_format='channels_first')
                    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=(2, 2),
                                             padding='same', activation=tf.nn.relu, name='encode_c2', data_format='channels_first')
                    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                             padding='same', activation=tf.nn.relu, name='encode_c3', data_format='channels_first')
                    nh = np.prod([v.value for v in conv3.get_shape()[1:]])
                    output_flat = tf.reshape(conv3, (-1, nh))
                    output_encode = tf.layers.dense(output_flat, 256, name='flatten_new', reuse=reuse)
                    output = tf.reshape(output_encode, (tf.shape(processed_x)[0], FLAGS.pred_steps, 256))
                    encode_lstm = tf.contrib.rnn.BasicLSTMCell(256, name="encode_lstm")

                    with tf.variable_scope("init", reuse=reuse):
                        start_state = tf.get_variable('init_state', shape=(1, 512), initializer=tf.initializers.zeros(), trainable=False)

                    encode_state = tf.split(tf.tile(start_state, (tf.shape(processed_x)[0], 1)), 2, axis=1)

                    for i in range(FLAGS.pred_steps-1, -1, -1):
                        output_i = output[:, i]
                        hidden, encode_state = encode_lstm(output_i, encode_state)
                        # encode_state = tf.squeeze(encode_state[0]), tf.squeeze(encode_state[1])

                    encode_state = tf.tile(hidden, (1, 5))

            elif FLAGS.i2a_action:

                def i2a_policy(inp, action=None, reuse=False, sample=False):
                    with tf.variable_scope("policy_copy", reuse=reuse):
                        conv1 = tf.layers.conv2d(inputs=inp, filters=32, kernel_size=[3, 3], strides=(8, 8),
                                                 padding='same', activation=tf.nn.relu, name='i2a_encode_c1', data_format='channels_first')
                        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=(2, 2),
                                                 padding='same', activation=tf.nn.relu, name='i2a_encode_c2', data_format='channels_first')
                        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                                 padding='same', activation=tf.nn.relu, name='i2a_encode_c3', data_format='channels_first')
                        nh = np.prod([v.value for v in conv3.get_shape()[1:]])
                        output_flat = tf.reshape(conv3, (-1, nh))
                        i2a_pd, _ = self.pdtype.pdfromlatent(output_flat)

                        if sample:
                            return i2a_pd.sample()
                        else:
                            return i2a_pd.logits

                batch = tf.shape(processed_x)
                processed_x_tile = tf.reshape(tf.tile(tf.reshape(processed_x, (batch[0], 1, batch[1], batch[2], batch[3])), (1, FLAGS.i2a_action_n, 1, 1, 1)), (batch[0]*FLAGS.i2a_action_n, 12, 84, 84))
                self.phys_model = PhysActionRollout(processed_x_tile, processed_x, [state], i2a_policy, FLAGS, env.action_space.n, ACTION, num_steps=FLAGS.pred_steps, reuse=reuse)
                final_output = self.phys_model.encode_output

                with tf.variable_scope("policy_model", reuse=reuse):

                    output = tf.reshape(tf.stop_gradient(final_output), (tf.shape(processed_x_tile)[0] * FLAGS.pred_steps, 3, 84, 84))

                    conv1 = tf.layers.conv2d(inputs=tf.stop_gradient(output), filters=32, kernel_size=[3, 3], strides=(8, 8),
                                             padding='same', activation=tf.nn.relu, name='encode_c1', data_format='channels_first')
                    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=(2, 2),
                                             padding='same', activation=tf.nn.relu, name='encode_c2', data_format='channels_first')
                    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                             padding='same', activation=tf.nn.relu, name='encode_c3', data_format='channels_first')


                    nh = np.prod([v.value for v in conv3.get_shape()[1:]])
                    output_flat = tf.reshape(conv3, (-1, nh))
                    output_encode = tf.layers.dense(output_flat, 256, name='flatten_new', reuse=reuse)
                    output = tf.reshape(output_encode, (tf.shape(processed_x_tile)[0], FLAGS.pred_steps, 256))
                    encode_lstm = tf.contrib.rnn.BasicLSTMCell(256, name="encode_lstm")

                    with tf.variable_scope("init", reuse=reuse):
                        start_state = tf.get_variable('init_state', shape=(1, 512), initializer=tf.initializers.zeros(), trainable=False)

                    encode_state = tf.split(tf.tile(start_state, (tf.shape(processed_x_tile)[0], 1)), 2, axis=1)

                    for i in range(FLAGS.pred_steps-1, -1, -1):
                        output_i = output[:, i]
                        hidden, encode_state = encode_lstm(output_i, encode_state)

                    encode_state = tf.reshape(hidden, (batch[0], FLAGS.i2a_action_n*256))



            else:
                self.phys_model = model_fn(processed_x, [state], FLAGS, num_steps=FLAGS.pred_steps, reuse=reuse)
                processed_x = tf.concat([processed_x, self.phys_model.final_output], axis=1)

                if FLAGS.balance_frames:
                    processed_x = processed_x[:, -12:]

            final_state = self.phys_model.final_state[0]


        if FLAGS.pm_rollouts:
            with tf.variable_scope("policy_model", reuse=reuse):
                self.phys_model = model_fn(processed_x, [state], FLAGS, num_steps=FLAGS.pred_steps, reuse=reuse)
                final_state = self.phys_model.final_state[0]
                reward_c1 = tf.layers.conv2d(inputs=tf.stop_gradient(self.phys_model.final_output), filters=32, kernel_size=[4, 4], strides=(2, 2), padding='same', activation=tf.nn.relu, name='reward_c1', data_format='channels_first', reuse=reuse)
                reward_c2 = tf.layers.conv2d(inputs=reward_c1, filters=32, kernel_size=[4, 4], strides=(2, 2), padding='same', activation=tf.nn.relu, name='reward_c2', data_format='channels_first', reuse=reuse)
                reward = tf.reduce_mean(tf.reduce_max(reward_c2, [2, 3]), [1])

        with tf.variable_scope("policy_model", reuse=reuse):

            if FLAGS.joint_policy:
                conv1 = tf.layers.conv2d(inputs=tf.stop_gradient(self.phys_model.conv_encode), filters=32, kernel_size=[3, 3], strides=(1, 1),
                                         padding='same', activation=tf.nn.relu, name='c1', data_format='channels_first')
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c2', data_format='channels_first')
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                         padding='same', activation=tf.nn.relu, name='c3', data_format='channels_first')
            else:
                conv1 = tf.layers.conv2d(inputs=tf.stop_gradient(processed_x), filters=32, kernel_size=[8, 8], strides=(4, 4),
                                         padding='same', activation=tf.nn.relu, name='c1', data_format='channels_first')
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c2', data_format='channels_first')
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                         padding='same', activation=tf.nn.relu, name='c3', data_format='channels_first')

            nh = np.prod([v.value for v in conv3.get_shape()[1:]])
            flat = tf.reshape(conv3, [-1, nh])

            if FLAGS.i2a or FLAGS.i2a_action:
                flat = tf.concat([flat, encode_state], axis=1)

            vf = tf.layers.dense(flat, 1, name='vf')[:, 0]
            flat = tf.nn.relu(flat)

            if FLAGS.order > 1:
                act_flat = tf.layers.dense(flat, actions*FLAGS.order)
                self.pd = self.pdtype.pdfromflat(act_flat)
            else:
                self.pd, _ = self.pdtype.pdfromlatent(flat)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        if FLAGS.dm_control:
            self.a0 = tf.math.tanh(a0)

        if FLAGS.i2a_action:
            with tf.variable_scope("conv_model_", reuse=reuse):
                i2a_logit = i2a_policy(processed_x, sample=False, reuse=True)
            policy_labels = tf.nn.softmax(self.pd.logits)
            self.policy_copy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(policy_labels), logits=i2a_logit)

        # Set value function prediction to not be influenced by future state predictions
        self.vf = vf

        if FLAGS.pm_rollouts:
            future_pred = tf.concat([processed_x, self.phys_model.final_output], axis=1)
            future_pred = future_pred[:, -12:]

            with tf.variable_scope("policy_model", reuse=True):

                conv1 = tf.layers.conv2d(inputs=tf.stop_gradient(future_pred), filters=32, kernel_size=[8, 8], strides=(4, 4),
                                         padding='same', activation=tf.nn.relu, name='c1', data_format='channels_first')
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c2', data_format='channels_first')
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1),
                                         padding='same', activation=tf.nn.relu, name='c3', data_format='channels_first')
                nh = np.prod([v.value for v in conv3.get_shape()[1:]])
                flat = tf.reshape(conv3, [-1, nh])

                if FLAGS.i2a:
                    flat = tf.concat([flat, encode_state], axis=1)

                future_vf = tf.layers.dense(flat, 1, name='vf')[:, 0]

            self.reward = reward
            self.future_vf = future_vf
            self.total_val = reward + (FLAGS.gamma ** 3) * future_vf

            vf = 0.1 * tf.maximum(self.total_val, vf) + 0.9 * vf

        def step(ob, state, done, ob_im=None):
            if FLAGS.pm_rollouts or FLAGS.pm:
                if FLAGS.dm_control:
                    return sess.run([a0, vf, final_state, neglogp0], {X:ob, STATE:state, MASK:done, X_IM:ob_im})
                else:
                    return sess.run([a0, vf, final_state, neglogp0], {X:ob, STATE:state, MASK:done})
            else:
                if ob_im is not None:
                    return sess.run([a0, vf, final_state, neglogp0], {X:ob, X_IM:ob_im})
                else:
                    return sess.run([a0, vf, final_state, neglogp0], {X:ob, X_IM:ob})

        def value(ob, state, done, ob_im=None):
            if FLAGS.pm_rollouts or FLAGS.pm:
                if FLAGS.dm_control:
                    return sess.run(vf, {X:ob, STATE:state, MASK:done, X_IM: ob_im})
                else:
                    return sess.run(vf, {X:ob, STATE:state, MASK:done})
            else:
                if ob_im is not None:
                    return sess.run(vf, {X:ob, X_IM: ob_im})
                else:
                    return sess.run(vf, {X:ob, X_IM: ob})


        self.step =step
        self.value = value
        self.final_state = final_state
        self.a0 = a0

        self.X = X
        self.X_IM = X_IM


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, gamma, lam, FLAGS):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.FLAGS = FLAGS

        if FLAGS.dm_control:
            self.obs_im = self.obs['pixels']
            self.obs_flat = self.obs['flat']

    def run(self):
        mb_obs, mb_actions, mb_neglogacs, mb_rewards, mb_dones, mb_values, mb_im_obs  = [], [], [], [], [], [], []
        mb_states = [self.states]
        epinfos = []

        FLAGS = self.FLAGS

        if FLAGS.dm_control:
            for _ in range(self.nsteps):
                actions, values, self.states, neglogacs = self.model.step(self.obs_flat, self.states, self.dones, self.obs_im)
                mb_obs.append(self.obs_flat.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogacs.append(neglogacs)
                mb_im_obs.append(self.obs_im.copy())
                mb_dones.append(self.dones)
                self.obs, rewards, self.dones, infos = self.env.step(actions)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)
                mb_rewards.append(rewards)
                mb_states.append(self.states)
                self.obs_im = self.obs['pixels']
                self.obs_flat = self.obs['flat']

            self.obs = self.obs_flat
        else:
            for _ in range(self.nsteps):
                actions, values, self.states, neglogacs = self.model.step(self.obs, self.states, self.dones)
                mb_obs.append(self.obs.copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogacs.append(neglogacs)
                mb_dones.append(self.dones)
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)
                mb_rewards.append(rewards)
                mb_states.append(self.states)

        mb_states = np.asarray(mb_states[:-1], dtype=np.float32)

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogacs = np.asarray(mb_neglogacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        if FLAGS.dm_control:
            last_value = self.model.value(self.obs, self.states, self.dones, self.obs_im)
        else:
            last_value = self.model.value(self.obs, self.states, self.dones)
            mb_im_obs = mb_obs

        mb_im_obs = np.asarray(mb_im_obs, dtype=np.uint8)

        # Now compute values and advantages
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)

        # Now compute future 3 reward and future 3 value
        mb_frs = np.zeros_like(mb_rewards)
        mb_fvs = np.zeros_like(mb_rewards)
        mb_valid = np.zeros_like(mb_rewards)

        mb_valid[:-3] = 1.

        last_gae_estimate = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                done = np.array(self.dones)
                td = mb_rewards[t] + self.gamma * last_value * (1 - done) - mb_values[t]
            else:
                done = mb_dones[t+1]
                td = mb_rewards[t] + self.gamma * mb_values[t+1] * (1 - done) - mb_values[t]

            mb_advs[t] = last_gae_estimate = td + self.gamma * self.lam * last_gae_estimate * (1 - done)

            if t < self.nsteps - 3:
                future_done = done

                for i in range(3):
                    future_done = (future_done * mb_dones[t+i])

                mb_fvs[t] = mb_returns[t+3] * future_done
                mb_frs[t] = mb_returns[t] - (self.gamma ** 3) * mb_fvs[t]

        mb_returns = mb_values + mb_advs

        # Flaten so time is on the second axises and batch is on the first
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogacs, mb_rewards, mb_states, mb_fvs, mb_frs, mb_valid, mb_im_obs)), epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def unflat(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.reshape(FLAGS.nenvs, FLAGS.nsteps, *s[1:])

def flat(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.reshape(s[0] * s[1], *s[2:])


def train(sess, act_model, train_model, env, resume_iter, logdir, FLAGS):
    nsteps = FLAGS.nsteps
    nminibatches = FLAGS.nminibatches
    ent_coef = FLAGS.ent_coeff
    vf_coef = FLAGS.vf_coef
    lr = FLAGS.lr
    exp = FLAGS.exp
    gamma = FLAGS.gamma
    lam = FLAGS.lam
    cliprate = FLAGS.cliprange
    env_id = FLAGS.env
    total_timesteps = int(1.1 * FLAGS.num_timesteps)

    # Set up training labels
    R = tf.placeholder(tf.float32, [None])
    A = train_model.pdtype.sample_placeholder([None])
    ADV = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRATE = tf.placeholder(tf.float32, [])
    OLDVPRED = tf.placeholder(tf.float32, [None])

    if FLAGS.dm_control:
        # PRED_LABEL = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
        PRED_LABEL = tf.placeholder(tf.float32, [None, 3*FLAGS.pred_steps, 84, 84])
    else:
        if FLAGS.i2a_action:
            PRED_LABEL = tf.placeholder(tf.float32, [None, 3, 84, 84])
        else:
            PRED_LABEL = tf.placeholder(tf.float32, [None, 3*FLAGS.pred_steps, 84, 84])

    # Set up *optimizers
    optim = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

    neglogac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    # For now don't use value function clip loss
    vf_sq1 = tf.square(train_model.vf - R)
    vf_loss = .5 * tf.reduce_mean(vf_sq1)

    ratio = tf.exp(OLDNEGLOGPAC - neglogac)
    pg_adv1= -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRATE, 1 + CLIPRATE)
    pg_adv2 = -ADV * ratio
    pg_loss = tf.reduce_mean(tf.maximum(pg_adv1, pg_adv2))

    loss = pg_loss - entropy * ent_coef + vf_coef * vf_loss

    copy_loss = tf.zeros(1)

    if FLAGS.i2a_action:
        copy_loss = tf.reduce_mean(train_model.policy_copy_loss)

    loss = loss + copy_loss


    if FLAGS.pm_rollouts:
        F_R = tf.placeholder(tf.float32, [None])
        F_V = tf.placeholder(tf.float32, [None])
        F_M = tf.placeholder(tf.float32, [None])
        vf_rollout_loss = .01 * (tf.reduce_mean(F_M * tf.square(train_model.reward - F_R)) + tf.reduce_mean(F_M * tf.square(train_model.future_vf - F_V)))
        loss = loss + vf_coef * vf_rollout_loss
    else:
        vf_rollout_loss = tf.zeros(1)

    # Not the actual KL since we don't have the probabilities of old actions
    approx_kl = .5 * tf.reduce_mean(tf.square(neglogac - OLDNEGLOGPAC))

    # Add losses to output
    params = tf.trainable_variables('policy_model')

    if FLAGS.i2a_action:
        params = params + tf.trainable_variables('conv_model_')

    grads = tf.gradients(loss, params)

    checks = []

    if FLAGS.joint_policy:
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        gvs = list(zip(grads, params))
        gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), v) for (grad, v) in gvs if grad is not None ]
        grads, params = zip(*gvs)
        capped_grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
        gvs = list(zip(capped_grads, params))
    else:
        capped_grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
        gvs = list(zip(capped_grads, params))

    _train = optim.apply_gradients(gvs)

    if FLAGS.pm or FLAGS.pm_rollouts:
        physics_params = tf.trainable_variables('conv_model_')
        loss_phys = tf.reduce_mean(tf.square(train_model.phys_model.final_output - PRED_LABEL))
        grads_phys = tf.gradients(loss_phys, physics_params)
        capped_grads, _ = tf.clip_by_global_norm(grads_phys, FLAGS.max_grad_norm)
        grads_phys = list(zip(capped_grads, physics_params))
        _train_phys = optim.apply_gradients(grads_phys)


    if FLAGS.resume_physics_only:
        loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="conv_model_"), max_to_keep=10)
        saver = tf.train.Saver(max_to_keep=10)
    elif FLAGS.resume_policy_only:
        loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_model"), max_to_keep=10)
        saver = tf.train.Saver(max_to_keep=10)
    else:
        saver = loader = tf.train.Saver(max_to_keep=10)

    if FLAGS.resume_iter != -1 or (not FLAGS.train):
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_iter = FLAGS.resume_iter
        try:
            loader.restore(sess, model_file)
        except:
            print("Mismatch between saved checked point and model parameters. Defaulting to optimistic saver. ")
            print("This will possibly lead to errors")
            optimistic_restore(sess, model_file)

    if FLAGS.resume_goal:
        model_file = osp.join(logdir, 'model_{}'.format(10000))
        optimistic_restore(sess, model_file)
    elif FLAGS.resume_forage:
        model_file = osp.join(logdir, 'model_{}'.format(10000))
        optimistic_restore(sess, model_file)

    def learn(lr, cliprate, obs, returns, masks, actions, values, neglogpacs, obs_im, states, *rollouts):
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {train_model.X: obs, LR:lr, CLIPRATE:cliprate, A:actions, ADV:advs, OLDNEGLOGPAC:neglogpacs,
                R: returns, OLDVPRED: values, train_model.X_IM: obs_im}

        if FLAGS.pm or FLAGS.pm_rollouts:
            td_map[train_model.STATE] = states
            td_map[train_model.MASK] = masks

        if FLAGS.pm_rollouts:
            f_v, f_r, f_m = rollouts
            td_map[F_V] = f_v
            td_map[F_R] = f_r
            td_map[F_M] = f_m

        try:
            return sess.run([pg_loss, vf_loss, vf_rollout_loss, entropy, approx_kl, copy_loss, _train], td_map)[:-1]
        except:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    if FLAGS.pm or FLAGS.pm_rollouts:
        def learn_phys(lr, label, obs, masks, states, obs_im, action):
            td_map = {train_model.X_IM: obs_im, PRED_LABEL:label, LR:lr, train_model.STATE:states, train_model.MASK:masks, train_model.ACTION: action}

            return sess.run([loss_phys, _train_phys], td_map)[:-1]

    loss_names = ['pg_loss', 'vf_loss', 'vf_rollout_loss', 'entropy', 'approx_kl', 'copy_loss']

    # Set up data collection
    runner = Runner(env, act_model, nsteps, gamma, lam, FLAGS)

    # Initialize Variables
    initialize()

    # Set up logger
    logger = TensorBoardOutputFormat(osp.join(FLAGS.logdir, FLAGS.exp))

    # Set up iterations of training
    total_iter = total_timesteps / nsteps
    nbatch = FLAGS.nenvs * nsteps
    nenvs = FLAGS.nenvs
    nbatch_train = nbatch // nminibatches
    nupdates = total_timesteps // nbatch
    noptsepochs = FLAGS.noptsepochs

    t_start = time.time()
    epinfobuf = deque(maxlen=100)

    if FLAGS.finetune_physics and FLAGS.mixed_finetune:
        data_finetune = np.load(FLAGS.finetune_data)['arr_0']
        length_finetune = data_finetune.shape[0]

        if FLAGS.env ==  'sonic':
            sonic_size = 2
            init_state = np.zeros((nbatch_train//sonic_size, 64, 21, 21))
            init_mask = np.zeros((nbatch_train//sonic_size,))
        else:
            init_state = np.zeros((nbatch_train//2, 64, 21, 21))
            init_mask = np.zeros((nbatch_train//2,))


    phys_loss, loss = 0, 0
    label_finetune = None

    for update in tqdm(range(resume_iter, nupdates)):
        obs, returns, masks, actions, values, neglogpacs, rewards, states, fvs, frs, fms, obs_im, epinfos  = runner.run()

        if not FLAGS.dm_control:
            obs_im = obs

        epinfobuf.extend(epinfos)
        mblossvals = []
        mblossphys = []

        frac = 1 - (update - 1.0) / nupdates

        if FLAGS.env == 'sonic' or FLAGS.dm_control:
            frac = 1.0

        lrnow = frac * lr
        clipratenow = frac * cliprate

        for _ in range(noptsepochs):
            inds = np.random.permutation(nbatch)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]

                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, obs_im))

                if FLAGS.pm or FLAGS.pm_rollouts:
                    mbstates = states[mbinds]
                else:
                    mbstates = None

                if FLAGS.pm_rollouts:
                    rollouts = (arr[mbinds] for arr in (fvs, frs, fms))
                else:
                    rollouts = []

                mblossvals.append(learn(lrnow, clipratenow, *slices, mbstates, *rollouts))

        lossvals = np.mean(mblossvals, axis=0)

        if FLAGS.finetune_physics:
            state_unflat = unflat(states)
            mask_unflat = unflat(masks)
            obs_im_unflat = unflat(obs_im)
            obs_unflat = unflat(obs_im)


            label = []

            state_unflat = unflat(states)[:, :-4]
            mask_unflat = unflat(masks)[:, :-4]
            action_unflat = unflat(actions)[:, :-4]
            obs_unflat = unflat(obs)
            label = obs_unflat[:, 4:, :, :, :9]
            obs_unflat = obs_unflat[:, :-4]

            state = flat(state_unflat)
            mask = flat(mask_unflat)
            obs = flat(obs_unflat)
            label = flat(label)
            actions = flat(action_unflat)

            delete_idx = np.where(mask>0)[0]
            valid_idx = []

            start_idx = 0

            for d_ix in delete_idx:
                end_idx = max(start_idx, d_ix - 3)
                valid_idx.extend(list(range(start_idx, end_idx)))
                start_idx = d_ix

            state = state[valid_idx]
            mask = mask[valid_idx]
            obs = obs[valid_idx]
            label = label[valid_idx]
            actions= actions[valid_idx]

            obs_im = obs

            assert not FLAGS.dm_control


            #################################

            if len(label.shape) == 4:
                label = np.transpose(label, axes=[0, 3, 1, 2]) / 255.

            if FLAGS.i2a_action:
                label = label[:, :3]


            if FLAGS.env == 'sonic':
                sonic_size = 2
                incre_size = nbatch_train // sonic_size
            else:
                incre_size = nbatch_train

            phys_nbatch = label.shape[0] - incre_size - 1

            if state.shape[0] > 0:
                for _ in range(FLAGS.finetune_epochs):
                    inds = np.random.permutation(phys_nbatch)
                    start = 0
                    for _ in range(FLAGS.nminibatches):
                        end = start + incre_size
                        mbinds = inds[start:end]

                        slices = list((arr[mbinds] for arr in (label, obs, masks, states, obs_im, actions)))
                        loss = learn_phys(FLAGS.phys_lr, *slices)[0]

                        start = end

                        if start > inds.shape[0]:
                            start = 0
                            inds = np.random.permutation(phys_nbatch)


        t_now = time.time()

        if update % FLAGS.log_interval == 0:
            summary = {}
            for (lossval, lossname) in zip(lossvals, loss_names):
                summary[lossname] = lossval

            if FLAGS.finetune_physics:
                if FLAGS.mixed_finetune:
                    summary['finetune_phys_loss'] = phys_loss
                summary['phys_loss'] = loss


            summary["explained_variance"] = explained_variance(values, returns)
            summary["return_mean"] =  returns.mean()
            summary["return_max"] =  returns.max()
            summary["return_min"] =  returns.min()
            summary["serial_timesteps"] = update*nsteps
            summary["total_timesteps"] = update*nbatch
            summary["nupdates"] = update
            summary["fps"] = update * nbatch / (t_now - t_start)
            summary["episode_reward_mean"] = safemean([epinfo['r'] for epinfo in epinfobuf])
            summary["episode_length_mean"] = safemean([epinfo['l'] for epinfo in epinfobuf])

            logger.writekvs(summary)

        if update % FLAGS.save_iter == 0:
            saver.save(sess, osp.join(logdir, 'model_{}'.format(update)))
            test_env = None

            if FLAGS.test_phys and FLAGS.pm:
                test_phys(sess, act_model, logdir, 'eval_{}.gif'.format(update), FLAGS, test_env)
            else:
                test(sess, act_model, logdir, 'eval_{}.gif'.format(update), FLAGS, test_env)
    env.close()
    return act_model

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def test(sess, act_model, logdir, name, FLAGS, env=None, load=False):
    save_path = osp.join(logdir, name)

    if env is None:
        env_id = FLAGS.env
        if FLAGS.env != 'phys_env' and FLAGS.env != 'phys_env_alternate' and FLAGS.env != 'phys_env_shooter' and FLAGS.env != "phys_env_3d" and FLAGS.env != "phys_env_3d_shooter":
            if FLAGS.dm_control:
                env = make_dm_control(env_id[0], env_id[1], 1, FLAGS.seed, 4, vis_reward=FLAGS.vis_reward)
            else:
                env = make_atari_env_custom(FLAGS.env, 1, FLAGS.seed+100, 4, random_action=FLAGS.random_action)
        else:
            def make_phys_env(rank):
                def _thunk():
                    if FLAGS.env == 'phys_env':
                        env = phys_env.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == 'phys_env_alternate':
                        env = phys_env_alternate.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == 'phys_env_shooter':
                        env = phys_env_shooter.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == "phys_env_3d":
                        env = phys_3d.PhysEnv3D(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == "phys_env_3d_shooter":
                        env = phys_3d_shooter.PhysEnv3D(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    env.seed(rank)
                    return env
                return _thunk

            env = SubprocVecEnv([make_phys_env(i) for i in range(1)])

    if load:
        loader = tf.train.Saver(max_to_keep=10)

        if FLAGS.resume_iter != -1 or (not FLAGS.train):
            model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
            resume_iter = FLAGS.resume_iter
            try:
                loader.restore(sess, model_file)
            except:
                print("Mismatch between saved checked point and model parameters. Defaulting to optimistic saver. ")
                optimistic_restore(sess, model_file)

    obs = env.reset()
    frames = []
    frames.append(obs[0, :, :, -3:])

    done = False
    states = act_model.initial_state[0:1]
    dones = [False for _ in range(1)]
    dones = np.array(dones)

    counter = 0

    while not done:
        actions, values, states, neglogacs = act_model.step(obs, states, dones)
        obs, rewards, dones, _ = env.step(actions)

        if FLAGS.pred_steps > 0 and (FLAGS.pm_oracle):
            idx = -(FLAGS.pred_steps + 1) * 3
            frames.append(obs[0, :, :, idx:idx+3])
        else:
            frames.append(obs[0, :, :, -3:])

        done = dones[0]

        counter += 1

        if counter > 100:
            break

    imageio.mimwrite(save_path, frames)


def test_phys(sess, act_model, logdir, name, FLAGS, env=None, load=False):
    save_path = osp.join(logdir, name)

    if env is None:
        env_id = FLAGS.env
        if FLAGS.env != 'phys_env' and FLAGS.env != 'phys_env_alternate' and FLAGS.env != 'phys_env_shooter' and FLAGS.env != "phys_env_3d" and FLAGS.env != "phys_env_3d_shooter":
            if FLAGS.dm_control:
                env = make_dm_control(env_id[0], env_id[1], 1, FLAGS.seed, 4, vis_reward=FLAGS.vis_reward)
            else:
                env = make_atari_env_custom(FLAGS.env, 1, FLAGS.seed+100, 4, random_action=FLAGS.random_action)
        else:
            def make_phys_env(rank):
                def _thunk():
                    if FLAGS.env == 'phys_env':
                        env = phys_env.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == 'phys_env_alternate':
                        env = phys_env_alternate.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == 'phys_env_shooter':
                        env = phys_env_shooter.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == "phys_env_3d":
                        env = phys_3d.PhysEnv3D(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    elif FLAGS.env == "phys_env_3d_shooter":
                        env = phys_3d_shooter.PhysEnv3D(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                               pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                    env.seed(rank)
                    return env
                return _thunk

            env = SubprocVecEnv([make_phys_env(i) for i in range(1)])

    assert FLAGS.pm

    if load:
        loader = tf.train.Saver(max_to_keep=10)

        if FLAGS.resume_iter != -1 or (not FLAGS.train):
            model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
            resume_iter = FLAGS.resume_iter
            try:
                loader.restore(sess, model_file)
            except:
                print("Mismatch between saved checked point and model parameters. Defaulting to optimistic saver. ")
                print("This will probably lead to errors")
                optimistic_restore(sess, model_file)

    obs = env.reset()
    frames = []
    frames.append(np.concatenate([obs[0, :, :, -3:], obs[0, :, :, -3:], obs[0, :, :, -3:], obs[0, :, :, -3:]], axis=1))

    done = False
    states = act_model.initial_state[0:1]
    dones = [False for _ in range(1)]
    dones = np.array(dones)

    counter = 0

    while not done:
        actions, states, output_frame = sess.run([act_model.a0, act_model.final_state, act_model.phys_model.final_output], {act_model.X:obs, act_model.STATE: states, act_model.MASK: dones})
        obs, rewards, dones, _ = env.step(actions)

        pred_frames = []

        for i in range(3):
            pred_frame = output_frame[0, i*3:(i+1)*3].transpose(1, 2, 0)
            pred_frame = (255 * np.clip(pred_frame, 0, 1)).astype(np.uint8)
            pred_frames.append(pred_frame)

        frames.append(np.concatenate([obs[0, :, :, -3:]] + pred_frames, axis=1))

        done = dones[0]

        counter += 1

        if counter > 100:
            break

    imageio.mimwrite(save_path, frames)


def atari_eval(FLAGS):
    env_id = FLAGS.env
    env = make_atari_env_custom(env_id, FLAGS.nenvs, FLAGS.seed, 4, random_action=FLAGS.random_action, augment=True)
    sess = tf.InteractiveSession()
    model_fn = PhysNetV2
    act_model = ConvPolicy(sess, env, FLAGS, model_fn, reuse=False)
    train_model = ConvPolicy(sess, env, FLAGS, model_fn, reuse=True)
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    exp_trim = FLAGS.exp.replace("_test_noise", "")
    command = "gsutil cp -r 'gs://yilundu-rcall/results/{}/model_10000.*' {}/".format(exp_trim, logdir)
    err = os.system(command)

    initialize()
    loader = tf.train.Saver(max_to_keep=10)
    model_file = osp.join(logdir, 'model_10000')
    loader.restore(sess, model_file)
    rewards = []
    obs = env.reset()
    states = act_model.initial_state

    while True:
        dones = [False for _ in range(1)]
        dones = np.array(dones)
        done = False

        actions, values, states, neglogacs = act_model.step(obs, states, dones)
        obs, reward, dones, infos = env.step(actions)

        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                rewards.append(maybeepinfo)
                print(len(rewards))

        if len(rewards) > 100:
            break

    rewards = np.array(rewards)
    np.save(osp.join(logdir, "rewards.npy"), rewards)

def main(FLAGS):
    env_id = FLAGS.env
    if FLAGS.env != 'phys_env' and FLAGS.env != 'phys_env_alternate' and FLAGS.env != 'phys_env_shooter' and FLAGS.env != "phys_env_3d" and FLAGS.env != "phys_env_3d_shooter":
        if FLAGS.dm_control:
            env = make_dm_control(env_id[0], env_id[1], FLAGS.nenvs, FLAGS.seed, 4, vis_reward=FLAGS.vis_reward)
        else:
            env = make_atari_env_custom(env_id, FLAGS.nenvs, FLAGS.seed, 4, random_action=FLAGS.random_action, augment=FLAGS.augment, eps_random=FLAGS.eps_random, random_fix=FLAGS.random_fix)
    else:
        def make_phys_env(rank):
            def _thunk():
                if FLAGS.env == 'phys_env':
                    env = phys_env.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                           pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                elif FLAGS.env == 'phys_env_alternate':
                    env = phys_env_alternate.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                           pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                elif FLAGS.env == 'phys_env_shooter':
                    env = phys_env_shooter.PhysEnv(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                           pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames, pos=FLAGS.pos)
                elif FLAGS.env == "phys_env_3d":
                    env = phys_3d.PhysEnv3D(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                           pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)
                elif FLAGS.env == "phys_env_3d_shooter":
                    env = phys_3d_shooter.PhysEnv3D(order=FLAGS.order, frame_stack=4, pm_oracle=FLAGS.pm_oracle,
                                           pred_steps=FLAGS.pred_steps, balance_frames=FLAGS.balance_frames)

                env.seed(rank)
                return env
            return _thunk

        env = SubprocVecEnv([make_phys_env(i+FLAGS.seed) for i in range(FLAGS.nenvs)])

    sess = tf.InteractiveSession()

    if FLAGS.model == 'physnet':
        model_fn = PhysNetV2
    else:
        model_fn = JunyukRecurrentNet

    if FLAGS.i2a_action:
        model_fn = PhysActionRollout

    act_model = ConvPolicy(sess, env, FLAGS, model_fn, reuse=False)
    train_model = ConvPolicy(sess, env, FLAGS, model_fn, reuse=True)

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    resume_iter = FLAGS.resume_iter

    # Initialize Variables
    initialize()

    if FLAGS.resume_goal:
        command = "gsutil cp -r 'gs://yilundu-rcall/results/phys_env_1_phys/model_10000.*' {}/".format(logdir)
        err = os.system(command)
    elif FLAGS.resume_forage:
        command = "gsutil cp -r 'gs://yilundu-rcall/results/phys_env_alternate_1_phys/model_10000.*' {}/".format(logdir)
        err = os.system(command)


    if (FLAGS.pm or FLAGS.pm_rollouts) and not FLAGS.pm_nova:
        phys_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="conv_model_"), max_to_keep=10)

        if FLAGS.blink:
            phys_model_file = pm_path_blink
        else:
            phys_model_file = FLAGS.pm_path

        if not FLAGS.i2a_action:
            phys_saver.restore(sess, phys_model_file)

    if FLAGS.train:
        train(sess, act_model, train_model, env, resume_iter, logdir, FLAGS)
    elif FLAGS.test_phys:
        test_phys(sess, act_model, logdir, 'eval_{}_phys.gif'.format(resume_iter), FLAGS, load=True)
    else:
        test(sess, act_model, logdir, 'eval_{}.gif'.format(resume_iter), FLAGS)


def batch_main():

    # Environments to run algorithm on
    envs = ["PongNoFrameskip-v0", "FrostbiteNoFrameskip-v0", "DemonAttackNoFrameskip-v0", "EnduroNoFrameskip-v0", "AsteroidsNoFrameskip-v0",
            "BreakoutNoFrameskip-v0", "FishingDerbyNoFrameskip-v0", "IceHockeyNoFrameskip-v0", "TennisNoFrameskip-v0", "AssaultNoFrameskip-v0",]

    # Physics Environments
    # envs = ["phys_env", "phys_env_alternate", "phys_env_shooter"]

    # Physics 3D Environments
    # envs = ["phys_env_3d_shooter", "phys_env_3d"]

    # Seeds to evaluate on
    seeds = [0, 2]

    for env in envs:
        for seed in seeds:
            # # # ##################################
            # Run experiment with IPA
            flag_tmp = EasyDict(FLAGS.copy())
            flag_tmp.pm = True
            flag_tmp.finetune_physics = True
            name = '{}_{}_ipa'.format(env, seed)
            flag_tmp.logdir = "/root/results"
            flag_tmp.exp = name
            flag_tmp.env = env
            flag_tmp.seed = seed

            main(flag_tmp)


            # # # ##################################
            # Run experiment with PPO
            flag_tmp = EasyDict(FLAGS.copy())
            flag_tmp.pm = False
            name = '{}_{}_ppo'.format(env, seed)
            flag_tmp.logdir = "/root/results"
            flag_tmp.exp = name
            flag_tmp.env = env
            flag_tmp.seed = seed

            main(flag_tmp)


if __name__ == '__main__':
    batch_main()
