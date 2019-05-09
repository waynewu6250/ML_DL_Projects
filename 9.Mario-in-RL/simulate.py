import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import EnvHandler
from models import Agent
from tqdm import trange
from pandas import DataFrame
ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values


def forward_run(is_new, agent=None, env_handler=None, game = 2):
    global_rewards = []

    if is_new:
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        env_handler = EnvHandler('SuperMarioBros2-v0', 1)
        agent = Agent("agent", env_handler.obs_shape, env_handler.n_actions)
    
    for i in range(game):

        total_rewards = 0
        done = 0
        if is_new:
            sess.run(tf.global_variables_initializer())

        while not done:
            action = agent.sample_actions(agent.step([env_handler.observation]))[0]
            env_handler.observation, reward, done, info = env_handler.env.step(action)
            total_rewards += reward
            
            env_handler.run()
            
            if done:
                env_handler.observation = env_handler.env.reset()
        
        global_rewards.append(total_rewards)

    # env_handler.render_rgb()
    # env_handler.render_frames(env_handler.observation)
    # env.close()

    return global_rewards

def train_agent(game = 10):
    env_handler = EnvHandler('SuperMarioBros2-v0', 10)
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    agent = Agent("agent", env_handler.obs_shape, env_handler.n_actions)
    sess.run(tf.global_variables_initializer())

    batch_states = env_handler.parallel_reset()

    rewards_history = []
    entropy_history = []

    for i in trange(100): 
        batch_actions = agent.sample_actions(agent.step(batch_states))
        batch_next_states, batch_rewards, batch_done, _ = env_handler.parallel_step(batch_actions)
        feed_dict = {
            agent.states_ph: batch_states,
            agent.actions_ph: batch_actions,
            agent.next_states_ph: batch_next_states,
            agent.rewards_ph: batch_rewards,
            agent.is_done_ph: batch_done,
        }
        batch_states = batch_next_states

        _, ent_t = sess.run(agent.train(), feed_dict)
        entropy_history.append(np.mean(ent_t))

        if (i+1) % 5 == 0:
            if (i+1) % 10 == 0:
                rewards_history.append(np.mean(forward_run(False, agent, env_handler, 1)))
                if rewards_history[-1] >= 50:
                    print("Your agent has earned the yellow belt: yah")
            print("Rewards: ",rewards_history)
            print("Current Entropy: ", entropy_history)
            # plt.figure(figsize=[8,4])
            # plt.subplot(1,2,1)
            # plt.plot(rewards_history, label='rewards')
            # plt.plot(ewma(np.array(rewards_history),span=10), marker='.', label='rewards ewma@10')
            # plt.title("Session rewards"); plt.grid(); plt.legend()
            
            # plt.subplot(1,2,2)
            # plt.plot(entropy_history, label='entropy')
            # plt.plot(ewma(np.array(entropy_history),span=1000), label='entropy ewma@1000')
            # plt.title("Policy entropy"); plt.grid(); plt.legend()        
            # plt.show()

if __name__ == "__main__":
    # game = 2
    # global_rewards = forward_run(True)
    # for i in range(game):
    #     print("Game Rewards: ", global_rewards[i])
    train_agent()

