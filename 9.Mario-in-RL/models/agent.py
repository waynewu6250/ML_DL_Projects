import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model, Sequential
import numpy as np

class Agent:
    def __init__(self, name, state_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""
        
        self.nn = Sequential([Conv2D(32, (3, 3), strides=(2,2), activation='relu', input_shape=state_shape),
                              Conv2D(32, (3, 3), strides=(2,2), activation='relu'),
                              Conv2D(32, (3, 3), strides=(2,2), activation='relu'),
                              Flatten(),
                              Dense(128, activation='relu')])
        
        with tf.variable_scope(name, reuse=reuse):
            
            # Prepare neural network architecture
            inputs = Input(shape=state_shape)
            x = self.nn(inputs)
            logits = Dense(n_actions, activation='linear')(x)
            state_value = Dense(1, activation='linear')(x)
            
            self.network = Model(inputs=inputs, outputs=[logits, state_value])
            
            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None,] + list(state_shape))
            self.agent_outputs = self.symbolic_step(self.state_t)
        
    def symbolic_step(self, state_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""
        
        # Apply neural network
        logits, state_value = self.network(state_t)
        state_value = tf.squeeze(state_value, axis=1)
        
        return (logits, state_value)
    
    def step(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.agent_outputs, {self.state_t: state_t})
    
    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_value = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])
