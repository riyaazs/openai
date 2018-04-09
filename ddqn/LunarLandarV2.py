import tensorflow as tf
from openai.ddqn import Agent

with tf.Session() as sess:
    agent = Agent.Agent(session=sess, train_episodes=5000, batch_size=64, beta0=0.4,
                        decay_rate=0.00001, learning_rate=0.0001, per_alpha = 0, per_epsilon= 1,
                        chk_pt_name="checkpoints/lunar_lander_ddqn_learning_rate_002.ckpt")
    agent.run_and_plt()
