"""Auxilary files for those who wanted to solve with policy gradient"""
import gym
from gym.core import Wrapper
from gym.spaces.box import Box
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

class EnvHandler:
    def __init__(self, game, n_envs):
        self.env, self.observation = self.make_env(game)
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.envs = [self.make_env(game)[0] for _ in range(n_envs)]


    def make_env(self, game):
        #All levels, starting at 1-1. With frameskip 4.
        env = gym_super_mario_bros.make(game)
        env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
        env = EnvWrapper(env, height=42, width=42,
                            crop = lambda img: img[60:-30, 5:],
                            dim_order = 'tensorflow',
                            color=False, n_frames=4,
                            reward_scale = 0.01)
        observation = env.reset()
        return env, observation
    
    def render_rgb(self):
        """ render rgb environment """
        plt.imshow(self.env.render('rgb_array'))
        plt.show()
    
    def render_frames(self, states):
        """ render processed frames """
        plt.imshow(states.transpose([0,2,1]).reshape([42,-1]))
        plt.show()

    # For forward run
    def run(self):
        """ run current environment """
        self.env.render(mode='human')
    
    # For parallel run
    def parallel_reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.array([env.reset() for env in self.envs])
    
    def parallel_step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done, infos = map(np.array, zip(*results))
        
        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()
        
        return new_obs, rewards, done, infos


# To wrap env into right height and width environment
class EnvWrapper(Wrapper):
    def __init__(self, env, height=42, width=42, color=False, crop=lambda img: img, 
                 n_frames=4, dim_order='theano', reward_scale=1,):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(EnvWrapper, self).__init__(env)
        assert dim_order in ('theano', 'tensorflow')
        self.img_size = (height, width)
        self.crop=crop
        self.color=color
        self.dim_order = dim_order
        self.reward_scale = reward_scale
        
        n_channels = (3 * n_frames) if color else n_frames
        obs_shape = [n_channels,height,width] if dim_order == 'theano' else [height,width,n_channels]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')
    
    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer
    
    def step(self,action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward * self.reward_scale, done, info
    
    ### image processing ###
    
    def update_buffer(self,img):
        img = self.preproc_image(img)
        offset = 3 if self.color else 1
        if self.dim_order == 'theano':
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset]
        else:
            axis = -1
            cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis = axis)

    def preproc_image(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if not self.color:
            img = img.mean(-1, keepdims=True)
        if self.dim_order == 'theano':
            img = img.transpose([2,0,1]) # [h, w, c] to [c, h, w]
        img = img.astype('float32') / 255.
        return img
