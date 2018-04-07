import tensorflow as tf
from openai.ddqn import Agent
import matplotlib.pyplot as plt

with tf.Session() as sess:
    agent = Agent.Agent(session=sess, train_episodes=5000, batch_size=64,
                        decay_rate=0.00001, learning_rate=0.0001,
                        chk_pt_name="checkpoints/lunar_lander_ddqn_learning_rate_002.ckpt")
    train_episodes_num_1, train_smoothed_reward_1, test_rewards_1 = agent.run()
    plt.plot(train_episodes_num_1, train_smoothed_reward_1,
             label="batch size=64, learning rate=0.0001, layers=2, hidden units=256 each")
    plt.title("Training stats with 100 episode moving average reward")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend(loc=4)
    plt.savefig("train.png")

    plt.clf()