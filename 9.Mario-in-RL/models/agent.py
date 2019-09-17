import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, Flatten, Input, LSTM, RepeatVector
from keras.models import Model, Sequential
import numpy as np
from config import opt

class Agent:
    def __init__(self, name, state_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""

        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.nn = Sequential([Conv2D(32, (3, 3), strides=(2,2), activation='relu', input_shape=state_shape),
                              Conv2D(32, (3, 3), strides=(2,2), activation='relu'),
                              Conv2D(32, (3, 3), strides=(2,2), activation='relu'),
                              Conv2D(32, (3, 3), strides=(2,2), activation='relu'),
                              Flatten(),
                              RepeatVector(1),
                              LSTM(512)])
        
        with tf.variable_scope(name, reuse=reuse):
            
            # Prepare neural network architecture
            inputs = Input(shape=state_shape)
            x = self.nn(inputs)
            logits = Dense(n_actions, activation='linear')(x)
            state_value = Dense(1, activation='linear')(x)
            
            self.network = Model(inputs=inputs, outputs=[logits, state_value])
            
            # prepare a graph for agent step
            self.states_ph = tf.placeholder('float32', [None,] + list(state_shape), name="states_ph")
            self.agent_outputs = self.symbolic_step(self.states_ph)

            self.next_states_ph = tf.placeholder('float32', [None,] + list(state_shape), name="next_states_ph")
            self.actions_ph = tf.placeholder('int32', (None,), name="actions_ph")
            self.rewards_ph = tf.placeholder('float32', (None,), name="rewards_ph")
            self.is_done_ph = tf.placeholder('float32', (None,), name="is_done_ph")

        
    def symbolic_step(self, state_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""
        
        # Apply neural network
        logits, state_value = self.network(state_t)
        state_value = tf.squeeze(state_value, axis=1)
        
        return (logits, state_value)
    
    def step(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.agent_outputs, {self.states_ph: state_t})
    
    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_value = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])
    
    def train(self):
        # logits[n_envs, n_actions] and state_values[n_envs, n_states]
        logits, state_values = self.symbolic_step(self.states_ph)
        next_logits, next_state_values = self.symbolic_step(self.next_states_ph)
        next_state_values = next_state_values * (1 - self.is_done_ph)

        # probabilities and log-probabilities for all actions
        probs = tf.nn.softmax(logits)            # [n_envs, n_actions]
        logprobs = tf.nn.log_softmax(logits)     # [n_envs, n_actions]

        # log-probabilities only for agent's chosen actions
        logp_actions = tf.reduce_sum(logprobs * tf.one_hot(self.actions_ph, self.n_actions), axis=-1) # [n_envs,]
        
        # compute advantage using rewards_ph, state_values and next_state_values
        advantage = (self.rewards_ph+opt.gamma*next_state_values)-state_values
        entropy = -tf.reduce_sum(probs*logprobs,axis=1)

        actor_loss =  - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - opt.beta * tf.reduce_mean(entropy)

        # compute target state values using temporal difference formula. Use rewards_ph and next_state_values
        target_state_values = self.rewards_ph + opt.gamma*next_state_values

        critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2)

        train_step = tf.train.AdamOptimizer(opt.lr).minimize(actor_loss + critic_loss)

        return (train_step, entropy)
