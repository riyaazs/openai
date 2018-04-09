import tensorflow as tf
import gym
import numpy as np
import time
from openai.ddqn import QNetwork
from openai.per import sumtree

class Agent:
    def __init__(self,
                 session,
                 max_steps_in_episode=1000,
                 train_episodes=5000,
                 gamma=0.99,
                 decay_rate=0.00001,
                 hidden_size=256,
                 learning_rate=0.00001,
                 update_freq = 100,
                 memory_size=100000,
                 batch_size=64,
                 layer_size = 2,
                 env_name = "LunarLander-v2",
                 chk_pt_name ="checkpoints/lunar_lander.ckpt"):
        self.session = session
        self.max_steps = max_steps_in_episode
        self.train_episodes = train_episodes
        self.gamma = gamma
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = decay_rate
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env.reset()
        self.mainQN = QNetwork.QNetwork(name='main',
                                        hidden_size=hidden_size,
                                        layer_size = layer_size,
                                        learning_rate=learning_rate,
                                        state_size = self.env.observation_space.shape[0],
                                        action_size = self.env.action_space.n)
        self.targetQN = QNetwork.QNetwork(name='target',
                                          hidden_size=hidden_size,
                                          layer_size=layer_size,
                                          learning_rate=learning_rate,
                                          state_size = self.env.observation_space.shape[0],
                                          action_size = self.env.action_space.n)
        self.memory = sumtree.SumTree(capacity=memory_size)
        self.saver = tf.train.Saver()
        self.chk_pt_name = chk_pt_name
        self.rewards_list = []
        self.test_rewards_list = []
        self.trainables = tf.trainable_variables()
        self.targetOps = self.updateTargetGraph(self.trainables, 1)

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    # Pretrain before experience replay buffer has minimum required data.
    def __pretrain(self):
        pretrain_length = self.batch_size
        # Make a bunch of random actions and store the experiences
        # Needed for experience replay.
        state, reward, done, _ = self.env.step(self.env.action_space.sample())
        for ii in range(pretrain_length):
            # Make a random action
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)

            if done:
                # The simulation fails so no next state
                next_state = np.zeros(self.env.observation_space.shape[0])
                # Add experience to memory
                self.memory.add(1.0, (state, action, reward, next_state))

                # Start new episode
                self.env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = self.env.step(self.env.action_space.sample())
            else:
                # Add experience to memory
                self.memory.add(1.0, (state, action, reward, next_state))
                state = next_state

    # Compute moving average of the rewards obtained in episodes.
    def __running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N

    def __train(self):
       # Initialize variables
        self.session.run(tf.global_variables_initializer())
        step = 0
        loss = None
        state, reward, done, _ = self.env.step(self.env.action_space.sample())
        for ep in range(1, self.train_episodes):
            total_reward = 0
            t = 0
            while t < self.max_steps:
                step += 1

                # Explore or Exploit
                explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = self.env.action_space.sample()
                else:
                    # Get action from Q-network
                    feed = {self.mainQN.inputs_: state.reshape((1,self.env.observation_space.shape[0]))}
                    Qs = self.session.run(self.mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(self.env.observation_space.shape[0])
                    t = self.max_steps

                    if loss != None:
                        print('Episode: {}'.format(ep),
                            'Total reward: {}'.format(total_reward),
                            'Training loss: {:.4f}'.format(loss),
                            'Explore P: {:.4f}'.format(explore_p))
                    self.rewards_list.append((ep, total_reward))
                    # Add experience to memory
                    self.memory.add(1.0, (state, action, reward, next_state))

                    # Start new episode
                    self.env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = self.env.step(self.env.action_space.sample())

                else:
                    # Add experience to memory
                    self.memory.add(1.0, (state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                exp_sample = self.memory.sample(self.batch_size)
                indices = exp_sample[0]
                batch = exp_sample[2]
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network, we alternate primary and target Qnetworks
                # for Double DQN.
                if ep < 100:
                    QN = self.mainQN
                else:
                    QN = self.targetQN

                current_Qs = self.session.run(QN.output, feed_dict={QN.inputs_: states})
                target_Qs = self.session.run(QN.output, feed_dict={QN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = np.zeros(self.env.action_space.n)

                targets = rewards + self.gamma * np.max(target_Qs, axis=1)


                loss, _ = self.session.run([self.mainQN.loss, self.mainQN.opt],
                               feed_dict={self.mainQN.inputs_: states,
                                          self.mainQN.targetQs_: targets,
                                          self.mainQN.actions_: actions})
                if ep%100==0 :
                    self.updateTarget(self.targetOps, self.session)
        self.saver.save(self.session, self.chk_pt_name)

    def test(self, test_episodes):
        self.saver.restore(self.session, self.chk_pt_name)

        state, reward, done, _ = self.env.step(self.env.action_space.sample())
        total_reward = reward
        for ep in range(1, test_episodes):
            t = 0
            while t < self.max_steps:
                # Use to see the agent running.
                #self.env.render()
                #sleep(0.01)  # Time in seconds.

                # Get action from Q-network
                feed = {self.mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = self.session.run(self.mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    self.test_rewards_list.append(total_reward)
                    t = self.max_steps
                    self.env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = self.env.step(self.env.action_space.sample())
                    total_reward = reward
                else:
                    state = next_state
                    t += 1

    # trains and tests agent on 100 episode.
    def __run(self):
        self.__pretrain()
        self.__train()
        self.test(100)
        eps, rews = np.array(self.rewards_list).T
        smoothed_train_rews = self.running_mean(rews, 100)
        smoothed_train_eps = eps[-len(smoothed_train_rews):]
        return (smoothed_train_eps, smoothed_train_rews, self.test_rewards_list)

    def run_plt(self):
        time_str = str(time.time())
        train_episodes_num_1, train_smoothed_reward_1, test_rewards_1 = self.__run()
        plt.plot(train_episodes_num_1, train_smoothed_reward_1,
                 label="batch size=64, learning rate=0.0001, layers=2, hidden units=256 each")
        plt.title("Training stats with 100 episode moving average reward")
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend(loc=4)
        trainfig = "train-" + time_str + ".png"
        plt.savefig(trainfig)

        plt.clf()
        plt.plot(range(1, len(test_rewards_1) + 1), test_rewards_1,
                 label="batch size=64, learning rate=0.0001 layers=2, hidden units=256 each")
        plt.title("Test run rewards per episode")
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend(loc=4)
        testfig = "test-" + time_str + ".png"
        plt.savefig(testfig)
