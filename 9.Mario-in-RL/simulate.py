import time
import tensorflow as tf
from utils import EnvHandler
from models import Agent



def forward_run(game = 2):
    
    env_handler = EnvHandler('SuperMarioBros2-v0', 1)
    global_rewards = []

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    for i in range(game):

        total_rewards = 0
        done = 0
        
        agent = Agent("agent", env_handler.obs_shape, env_handler.n_actions)
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

def parallel_run(game = 10):
    env_handler = EnvHandler('SuperMarioBros2-v0', 10)
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    agent = Agent("agent", env_handler.obs_shape, env_handler.n_actions)
    sess.run(tf.global_variables_initializer())

    batch_states = env_handler.parallel_reset()
    batch_actions = agent.sample_actions(agent.step(batch_states))
    batch_next_states, batch_rewards, batch_done, _ = env_handler.parallel_step(batch_actions)

    print("State shape:", batch_states.shape)
    print("Actions:", batch_actions)
    print("Rewards:", batch_rewards)
    print("Done:", batch_done)


if __name__ == "__main__":
    game = 2
    global_rewards = forward_run(game)
    for i in range(game):
        print("Game Rewards: ", global_rewards[i])

