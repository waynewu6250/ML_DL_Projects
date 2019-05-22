import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import EnvHandler
from models import Agent
from tqdm import trange
from pandas import DataFrame
ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values
checkpoint_path = None #'checkpoints/model-19999.meta'


def forward_run(is_new, agent=None, env_handler=None, game = 2, game_name = 'SuperMarioBros-v0'):
    global_rewards = []
    if is_new:
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        env_handler = EnvHandler(game_name, 1)
        agent = Agent("agent", env_handler.obs_shape, env_handler.n_actions)

        if checkpoint_path:
            saver = tf.train.import_meta_graph(checkpoint_path)
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    
    for i in range(game):

        total_rewards = 0
        life = 2
        if is_new:
            sess.run(tf.global_variables_initializer())

        env_handler.observation = env_handler.env.reset()

        while True:
            
            action = agent.sample_actions(agent.step([env_handler.observation]))[0]
            env_handler.observation, reward, done, info = env_handler.env.step(action)
            total_rewards += reward
            life = info["life"]
            time = info["time"]
            
            env_handler.run()

            if life == 1 or time == 300 or done:
                env_handler.observation = env_handler.env.reset()
                break
        
        global_rewards.append(total_rewards)
        print("Game {} reward: {:.3f}".format(i+1,total_rewards))
    
    return global_rewards

    # env_handler.render_rgb()
    # env_handler.render_frames(env_handler.observation)
    # env.close()

def train_agent(game = 10, game_name = 'SuperMarioBros-v0'):
    env_handler = EnvHandler(game_name, 10)
    
    # Create Graph and agent
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    agent = Agent("agent", env_handler.obs_shape, env_handler.n_actions)
    train_step, entropy = agent.train()
    sess.run(tf.global_variables_initializer())
    
    if checkpoint_path:
        saver = tf.train.import_meta_graph(checkpoint_path)
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
    else:
        saver = tf.train.Saver()

    # Initialize parameters
    batch_states = env_handler.parallel_reset()
    entropy_history = []
    rewards_history = []


    # Create Figure
    plt.figure(figsize=(8,4), dpi=80)
    plt.ion()

    for i in trange(20000): 
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

        _, ent_t = sess.run([train_step, entropy], feed_dict)
        entropy_history.append(np.mean(ent_t))

        if (i+1) % 10 == 0:
            if (i+1) % 500 == 0:
                rewards_history.append(np.mean(forward_run(False, agent, env_handler, 1, game_name)))
                saver.save(sess, "checkpoints/model", global_step=i)
                if rewards_history[-1] >= 15:
                    print("Your agent has earned the yellow belt: yah")
                    print("Rewards: ",rewards_history[-1])

            plt.cla()
            plt.subplot(1,2,1)
            if rewards_history != []:
                plt.cla()
                plt.plot(rewards_history, label='rewards')
                plt.plot(ewma(np.array(rewards_history),span=10), marker='.', label='rewards ewma@1000')
                plt.title("Session rewards"); plt.grid(); plt.legend()
            
            plt.subplot(1,2,2)
            plt.plot(entropy_history, label='entropy')
            plt.plot(ewma(np.array(entropy_history),span=1000), label='entropy ewma@1000')
            plt.title("Policy entropy"); plt.grid(); plt.legend()

            plt.pause(0.1)     
    
    plt.ioff()
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="help", type=str, dest="mode", default="play")
    parser.add_argument("-n", "--number", help="number of games", type=int, dest="game", default=1)
    args = parser.parse_args()
    
    game_name = "SuperMarioBros-v0"

    if args.mode == "play":
        global_rewards = forward_run(True, game=args.game, game_name=game_name)
    elif args.mode == "train":
        train_agent(game=args.game, game_name=game_name)
    else:
        raise("Please only insert play or train in mode!!")

