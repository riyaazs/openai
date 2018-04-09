import tensorflow as tf
from openai.ddqn import Agent
import matplotlib.pyplot as plt

with tf.Session() as sess:
    agent = Agent.Agent(session=sess,
                        train_episodes=5000,
                        batch_size=64,
                        decay_rate=0.00001,
                        learning_rate=0.0001,
                        layer_size=2,
                        chk_pt_name="checkpoints/lunar_lander_ddqn_learning_rate_002.ckpt")
    agent.run_plt()
