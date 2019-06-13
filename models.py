import tensorflow as tf
import numpy as np
from utils import spatial_mem, residual_block, convlstm
# FLAGS = flags.FLAGS

class JunyukRecurrentNet(object):
    def __init__(self, OBS, STATE, FLAGS, scope="", num_channels=12, reuse=False, nlstm=1024, num_steps=10, self_predict=False, label=None):
        self.num_channels = num_channels
        self.STATE = tf.split(STATE[0], 2, axis=1)
        batch_size = tf.shape(OBS)[0]

        """Otherwise use a autoregressive model"""
        self.OBS = OBS

        with tf.variable_scope("conv_model_"+scope, reuse=reuse):
            input_val = self.OBS
            outputs = []
            state = self.STATE
            lstm = tf.contrib.rnn.BasicLSTMCell(nlstm, name='prod_lstm')

            for i in range(num_steps):
                conv1 = tf.layers.conv2d(inputs=input_val, filters=64, kernel_size=(8, 8), strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c1',
                                         data_format='channels_first', reuse=reuse)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=(6,6), strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c2',
                                         data_format='channels_first', reuse=reuse)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=(4,4), strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c3',
                                         data_format='channels_first', reuse=reuse)
                nh = np.prod([v.value for v in conv3.get_shape()[1:]])

                flat = tf.reshape(conv3, [batch_size, nh])

                fc1 = tf.layers.dense(flat, 1024, name='pre_rec_fc', activation=tf.nn.relu, reuse=reuse)

                output, state = lstm(fc1, state)

                if i == 0:
                    state_next = state

                output = tf.layers.dense(output, 128 * 11 * 11, reuse=reuse, name='deconv_dense')
                output = tf.reshape(output, (batch_size, 128, 11, 11))

                deconv1 = tf.layers.conv2d_transpose(output, 128, kernel_size=(4,4), strides=(2, 2),
                                                     padding='same', activation=tf.nn.relu, name='deconv1',
                                                     data_format='channels_first', reuse=reuse)
                deconv2 = tf.layers.conv2d_transpose(deconv1, 128, kernel_size=(6,6), strides=(2, 2),
                                                     padding='same', activation=tf.nn.relu, name='deconv2',
                                                     data_format='channels_first', reuse=reuse)
                deconv3 = tf.layers.conv2d_transpose(deconv2, 3, kernel_size=(8,8), strides=(2, 2),
                                                     padding='same', activation=tf.nn.relu, name='deconv3',
                                                     data_format='channels_first', reuse=reuse)

                new_input_val = deconv3[:, :, 2:-2, 2:-2]
                outputs.append(new_input_val)
                input_val = tf.concat([input_val[:, 3:, :, :], new_input_val], axis=1)
                reuse = True

                final_output = tf.stack(outputs, axis=1)

            final_output = tf.reshape(final_output, (tf.shape(final_output)[0], num_steps*3, 84, 84))
            self.final_output = final_output
            self.final_state = [tf.concat(state_next, axis=1)]


class PhysNetV2(object):
    def __init__(self, OBS, STATE, FLAGS, scope="", num_channels=12, reuse=False, num_steps=10, self_predict=False, label=None, bypass_res=False, training=False):
        self.num_channels = num_channels
        self.STATE = STATE
        batch_size = tf.shape(OBS)[0]

        """Otherwise use a autoregressive model"""
        self.OBS = OBS

        with tf.variable_scope("conv_model_"+scope, reuse=reuse):
            input_val = self.OBS
            outputs = []
            state = self.STATE
            states = []

            for i in range(num_steps):
                inp = conv1 = tf.layers.conv2d(inputs=input_val, filters=64, kernel_size=(8, 8), strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c1',
                                         data_format='channels_first', reuse=reuse)

                inp = residual_block(inp, 64, training, 'channels_first', name='rb_{}'.format(0), reuse=reuse)
                inp = tf.layers.conv2d(inputs=inp, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                       padding='same', activation=tf.nn.leaky_relu, name='c2', data_format='channels_first', reuse=reuse)
                inp = residual_block(inp, 64, training, 'channels_first', name='rb_{}'.format(1), reuse=reuse)

                inp = tf.reshape(inp, [batch_size, 64, 21, 21])

                if i == 0:
                    self.conv_encode = inp

                if FLAGS.convlstm:
                    state, output, d_state = convlstm(inp, state, FLAGS, reuse=reuse, bypass_res=bypass_res)
                else:
                    state, output, d_state = spatial_mem(inp, state, FLAGS, reuse=reuse, bypass_res=bypass_res)

                states.append(state)

                if i == 0:
                    state_next = state

                output = tf.reshape(output, (batch_size, 64, 21, 21))

                output = residual_block(output, 64, training, 'channels_first', name='drb_{}'.format(0), reuse=reuse)

                output = tf.layers.conv2d_transpose(output, 64, kernel_size=(4,4), strides=(2, 2),
                                                    padding='same', name='dc1',
                                                    data_format='channels_first', reuse=reuse, activation=tf.nn.relu)

                output = residual_block(output, 64, training, 'channels_first', name='drb_{}'.format(1), reuse=reuse)
                output = tf.layers.conv2d_transpose(output, 3, kernel_size=(8,8), strides=(2, 2),
                                                    padding='same', name='output_conv',
                                                    data_format='channels_first', reuse=reuse, activation=tf.nn.relu)

                new_input_val = tf.reshape(output, (batch_size, 3, 84, 84))
                outputs.append(new_input_val[:, :3, :, :])
                input_val = tf.concat([input_val[:, 3:, :, :], new_input_val[:, :3, :, :]], axis=1)
                input_val = tf.clip_by_value(input_val, 0, 1)
                reuse = True

            final_output = tf.stack(outputs, axis=1)

        final_output = tf.reshape(final_output, (tf.shape(final_output)[0], num_steps*3, 84, 84))
        self.final_output = final_output
        self.final_state = state_next
        self.states = states
        self.d_state = d_state


class PhysActionRollout(object):
    def __init__(self, OBS, OBS_ORIG, STATE, policy_fn, FLAGS, n, ACTION, scope="", num_channels=12, reuse=False, num_steps=10, self_predict=False, label=None, bypass_res=False, training=False):
        self.num_channels = num_channels
        STATE_ORIG  = STATE
        STATE = STATE[0]
        self.STATE = STATE
        self.OBS = OBS

        orig_state_shape = tf.shape(STATE)
        batch_size = tf.shape(OBS)[0]

        STATE = [tf.reshape(tf.tile(tf.reshape(STATE, (tf.shape(STATE)[0], 1, 64, 21, 21)), (1, FLAGS.i2a_action_n, 1, 1, 1)), (FLAGS.i2a_action_n*tf.shape(STATE)[0], 64, 21, 21))]
        with tf.variable_scope("conv_model_"+scope, reuse=reuse):
            input_val = self.OBS
            outputs = []
            state = STATE
            states = []

            for i in range(num_steps):
                inp = conv1 = tf.layers.conv2d(inputs=input_val, filters=64, kernel_size=(8, 8), strides=(2, 2),
                                         padding='same', activation=tf.nn.relu, name='c1',
                                         data_format='channels_first', reuse=reuse)

                inp = residual_block(inp, 64, training, 'channels_first', name='rb_{}'.format(0), reuse=reuse)
                inp = tf.layers.conv2d(inputs=inp, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                       padding='same', activation=tf.nn.leaky_relu, name='c2', data_format='channels_first', reuse=reuse)
                inp = residual_block(inp, 64, training, 'channels_first', name='rb_{}'.format(1), reuse=reuse)

                inp = tf.reshape(inp, [batch_size, 64, 21, 21])

                action = policy_fn(OBS, sample=True, reuse=bool(i))
                action = tf.gather(tf.eye(n), tf.squeeze(action), axis=0)
                action = tf.tile(tf.reshape(action, (tf.shape(action)[0], n, 1, 1)), (1, 1, 21, 21))
                action = tf.stop_gradient(action)

                state, output, d_state = spatial_mem(inp, state, FLAGS, reuse=reuse, bypass_res=bypass_res, action=action)

                if i == 0:
                    state_next = state

                output = tf.reshape(output, (batch_size, 64, 21, 21))

                output = residual_block(output, 64, training, 'channels_first', name='drb_{}'.format(0), reuse=reuse)

                output = tf.layers.conv2d_transpose(output, 64, kernel_size=(4,4), strides=(2, 2),
                                                    padding='same', name='dc1',
                                                    data_format='channels_first', reuse=reuse, activation=tf.nn.relu)

                output = residual_block(output, 64, training, 'channels_first', name='drb_{}'.format(1), reuse=reuse)
                output = tf.layers.conv2d_transpose(output, 3, kernel_size=(8,8), strides=(2, 2),
                                                    padding='same', name='output_conv',
                                                    data_format='channels_first', reuse=reuse, activation=tf.nn.relu)

                new_input_val = tf.reshape(output, (batch_size, 3, 84, 84))
                outputs.append(new_input_val[:, :3, :, :])
                input_val = tf.concat([input_val[:, 3:, :, :], new_input_val[:, :3, :, :]], axis=1)
                input_val = tf.clip_by_value(input_val, 0, 1)
                reuse = True

            final_output = tf.stack(outputs, axis=1)

        final_output = tf.reshape(final_output, (tf.shape(final_output)[0], num_steps*3, 84, 84))
        self.encode_output = final_output

        state_next = tf.reshape(state_next[0], (orig_state_shape[0], FLAGS.i2a_action_n, 64, 21, 21))
        state_next = state_next[:, 0, :, :, :]
        self.final_state = [state_next]


        with tf.variable_scope("conv_model_"+scope, reuse=True):
            input_val = OBS_ORIG
            outputs = []
            state = STATE_ORIG
            batch_size = tf.shape(OBS_ORIG)[0]

            inp = conv1 = tf.layers.conv2d(inputs=input_val, filters=64, kernel_size=(8, 8), strides=(2, 2),
                                     padding='same', activation=tf.nn.relu, name='c1',
                                     data_format='channels_first', reuse=reuse)

            inp = residual_block(inp, 64, training, 'channels_first', name='rb_{}'.format(0), reuse=reuse)
            inp = tf.layers.conv2d(inputs=inp, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                   padding='same', activation=tf.nn.leaky_relu, name='c2', data_format='channels_first', reuse=reuse)
            inp = residual_block(inp, 64, training, 'channels_first', name='rb_{}'.format(1), reuse=reuse)

            inp = tf.reshape(inp, [batch_size, 64, 21, 21])

            action = tf.gather(tf.eye(n), ACTION, axis=0)
            action = tf.tile(tf.reshape(action, (tf.shape(action)[0], n, 1, 1)), (1, 1, 21, 21))
            state, output, d_state = spatial_mem(inp, state, FLAGS, reuse=reuse, bypass_res=bypass_res, action=action)

            output = tf.reshape(output, (batch_size, 64, 21, 21))

            output = residual_block(output, 64, training, 'channels_first', name='drb_{}'.format(0), reuse=reuse)

            output = tf.layers.conv2d_transpose(output, 64, kernel_size=(4,4), strides=(2, 2),
                                                padding='same', name='dc1',
                                                data_format='channels_first', reuse=reuse, activation=tf.nn.relu)

            output = residual_block(output, 64, training, 'channels_first', name='drb_{}'.format(1), reuse=reuse)
            output = tf.layers.conv2d_transpose(output, 3, kernel_size=(8,8), strides=(2, 2),
                                                padding='same', name='output_conv',
                                                data_format='channels_first', reuse=reuse, activation=tf.nn.relu)

            new_input_val = tf.reshape(output, (batch_size, 3, 84, 84))

        self.final_output = new_input_val
