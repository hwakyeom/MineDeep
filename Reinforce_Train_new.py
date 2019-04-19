import sys
import numpy as np
import random
from collections import deque
#---------------------------------------
import gym
from gym.envs.registration import register
#---------------------------------------
import tensorflow as tf
#---------------------------------------
import pygame
#---------------------------------------



class Game:
    def __init__(self, name='NoName', screen_width=500, screen_height=300):
        pygame.init()
        pygame.display.set_caption(name)
        self._screen = pygame.display.set_mode((screen_width, screen_height))
        self._clock = pygame.time.Clock()
        self.Env = None
        self.Events = {}

        self.sess = tf.Session()

    def Register(self, id):
        #register(id=id, kwargs)
        self.Env = gym.make(id)
        return self.Env

    def InputAgent(self, cls_Agent):
        if self.Env is not None:
            cls_Agent.sess = self.sess
            cls_Agent.Env_ = self.Env
            cls_Agent.now_state = self.Env.reset()
            cls_Agent.states_num = self.Env.observation_space.n
            cls_Agent.action_num = self.Env.action_space.n
            cls_Agent.now_reward = 0
            cls_Agent.Dying = False
            cls_Agent.info = None
            cls_Agent.InitDQN()


    def GetEnv(self):
        return self.Env

    def Loop_Events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Close Button => Game Quit
                self.Quit()
            if event.type == pygame.KEYDOWN:    # in this game, only use keyboard
                Events_ = self.Events   # ex:) { ord('t') : [ minsu.train, 2000 ] }
                for key_name, function_iter in zip(Events_.keys(), Events_.values()):
                    if event.key == key_name:
                        try:
                            dofunc_ptr = function_iter[0](*function_iter[1:])
                        except Exception as e:
                            print('except ', e)


    def Quit(self):
        pygame.quit()
        sys.exit()


    def CreateCommand(self, command={ pygame.K_ESCAPE : Quit }):
        self.Events.update(command)


    def Render(self):
        self.Env.render()




class DQN:
    def __init__(self, _session, input_size, output_size, name_str='main', h_layer_size=10, learning_rate=0.01, drop_rate=0.7):
        self.sess = _session
        self.Name = name_str
        self.InputSize = input_size
        self.OutputSize = output_size
        self.HiddenLayerSize = h_layer_size
        self.LearningRate = learning_rate
        self.DropRate = drop_rate
        self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.Name):
            # ------
            He_init = tf.contrib.layers.variance_scaling_initializer()  # He init
            self.NowTraining = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, self.InputSize], name='Input_X')

            dense1 = tf.layers.dense(inputs=self.X, units=self.HiddenLayerSize, activation=tf.nn.relu, kernel_initializer=He_init)
            dense1_dropout = tf.layers.dropout(inputs=dense1, rate=self.DropRate, training=self.NowTraining)
            dense2 = tf.layers.dense(inputs=dense1_dropout, units=self.HiddenLayerSize, activation=tf.nn.relu, kernel_initializer=He_init)
            dense2_dropout = tf.layers.dropout(inputs=dense2, rate=self.DropRate, training=self.NowTraining)

            self.Predict = tf.layers.dense(inputs=dense2_dropout, units=self.OutputSize, kernel_initializer=He_init) # conventional

            # ------

        self.Y = tf.placeholder(tf.float32, [None, self.OutputSize], name='Real_Y')

        self.Cost = tf.reduce_mean(tf.square(self.Predict - self.Y))
        self.Train = tf.train.AdamOptimizer(self.LearningRate).minimize(self.Cost)

    def Qpredict(self, x):
        return self.sess.run([self.Predict], feed_dict={self.X: x, self.NowTraining: False})

    def Train(self, x, y):
        return self.sess.run([self.Cost, self.Train], feed_dict={self.X: x, self.Y: y, self.NowTraining: True})

    def Assign(self, Another_Q):
        op_holder = []
        dest_scope_name = self.Name
        src_scope_name = Another_Q.Name

        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        self.sess.run(op_holder) # assign assign assign




class Agent:
    def __init__(self, name_str):
        self.sess = None
        self.Env_ = None
        self.Name = name_str
        self.states_num = 0
        self.action_num = 0
        #---------------------------------------
        self.now_state = 0
        self.now_reward = 0
        self.Dying = False
        self.info = None
        self.Actions = {}

    def InitDQN(self):
        if self.sess is not None:
            self.MainQ = DQN(self.sess, self.states_num, self.action_num, 'Main')
            self.TargetQ = DQN(self.sess, self.states_num, self.action_num, 'Target')
            self.sess.run(tf.global_variables_initializer())
            self.TargetQ.Assign(self.MainQ)  # same same init


    def SetActions(self, _kwargs):    # ex:) { 'LEFT' : 0, 'DOWN' : 1, 'RIGHT' : 2, 'UP' : 3 }
        self.Actions = _kwargs

    def AddActions(self, _kwargs):
        self.Actions.update(_kwargs)

    def DoActionforActName(self, act_str):
        if self.Env_ is not None:
            self.now_state, self.now_reward, self.Dying, self.info = self.Env_.step(self.Actions[act_str])
            self.PrintInfo()

    def DoAction(self, act):
        if self.Env_ is not None:
            self.now_state, self.now_reward, self.Dying, self.info = self.Env_.step(act)


    def Reset(self):
        if self.Env_ is not None:
            self.Dying = False
            self.now_state = self.Env_.reset()

    def PrintInfo(self):
        print("State: ", self.now_state, "Reward: ", self.now_reward, "Info: ", self.info)



    # Train Mode
    def one_hot_for_state(self, x):
        return np.identity(self.MainQ.InputSize)[x:x+1]


    def _replay_train(self, train_batch, dis=0.99):
        x_stack = np.empty(0).reshape(0, self.MainQ.InputSize)
        y_stack = np.empty(0).reshape(0, self.MainQ.OutputSize)

        for state, action, reward, next_state, dying in train_batch:
            Q = self.MainQ.Qpredict(state)
            
            # if terminal
            print(Q)
            if reward == 1:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + dis * np.max(self.TargetQ.Qpredict(next_state))

            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state])

        return self.MainQ.Train(x_stack, y_stack)



    def SelfTrain(self, episode_num):
        Rewards_list = []
        Replay_buffer = deque()
        REPLAY_MEMORY = 5000

        for episode in range(episode_num):
            e = 1. / ((episode / 10) + 1)
            self.Reset()

            reward_All = 0
            step_count = 0

            while not self.Dying:
                if np.random.rand(1) < e:
                    action = np.random.choice(np.array(list(self.Actions.values())), 1)[0]
                else:
                    action = np.argmax(self.MainQ.Qpredict(self.one_hot_for_state(self.now_state)))

                prev_state = self.one_hot_for_state(self.now_state)
                self.now_state, self.now_reward, self.Dying, self.info = self.Env_.step(action)

                Replay_buffer.append((prev_state, action, self.now_reward, self.one_hot_for_state(self.now_state), self.Dying))
                if len(Replay_buffer) > REPLAY_MEMORY:
                    Replay_buffer.popleft()

                step_count += 1

                reward_All += self.now_reward

            Rewards_list.append(reward_All)
            print("Episode: {}  step: {}".format(episode, step_count))

            if episode % 10 == 1:
                for i in range(50):

                    if len(Replay_buffer) > 10:
                        minibatch = random.sample(Replay_buffer, 10)
                        cost, _ = self._replay_train(minibatch)


                #print("cost: ", cost)
                    
                self.TargetQ.Assign(self.MainQ)


        print("Sucess rate: " + str(sum(Rewards_list)/count))






Reinforce = Game('Reinforce Man', 300, 200)
Reinforce.Register('FrozenLake-v0')

player1 = Agent('minsu')
player1.SetActions({ 'LEFT' : 0, 'DOWN' : 1, 'RIGHT' : 2, 'UP' : 3 })

Reinforce.InputAgent(player1)


agent = player1
# ************************************************
def ActionAndRender(act_name_str):
    agent.DoAction(act_name_str)
    if agent.Dying:
        print('He\'s dead.')
        agent.Reset()
    Reinforce.Render()

#-------------------------------------------------
Reinforce.CreateCommand({
    ord('a'): [ActionAndRender, 'LEFT'],
    ord('d'): [ActionAndRender, 'RIGHT'],
    ord('w'): [ActionAndRender, 'UP'],
    ord('s'): [ActionAndRender, 'DOWN'],
})
# ************************************************




player1.SelfTrain(1000)

Reinforce.Render()

while True:

    Reinforce.Loop_Events()   # If you're simsim Just Do it


