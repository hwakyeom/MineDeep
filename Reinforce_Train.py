import sys
import numpy as np
import random
#---------------------------------------
import gym
from gym.envs.registration import register
#---------------------------------------
import pygame
#---------------------------------------



class Game:
    def __init__(self, name='NoName', screen_width=500, screen_height=300):
        pygame.init()
        pygame.display.set_caption(name)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.Now_Env = None
        self.Commands = {}

    def Register(self, id, entry_point, kwargs):
        register(id=id, entry_point=entry_point, kwargs=kwargs)
        self.Now_Env = gym.make(id)
        return self.Now_Env


    def GetEnv(self):
        return self.Now_Env

    def Loop_Events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Close Button => Game Quit
                self.Quit()
            if event.type == pygame.KEYDOWN:
                keys_ = list(self.Commands.keys())
                values_ = list(self.Commands.values())
                for key_type, func_tion in zip(keys_, values_):
                    if event.key == key_type:
                        try:
                            func_pointer = func_tion[0](*func_tion[1:])
                        except Exception as e:
                            print('except ', e)


    def Quit(self):
        pygame.quit()
        sys.exit()

    def CreateCommand(self, command={ pygame.K_ESCAPE : Quit }):
        self.Commands.update(command)


    def Render(self):
        self.Now_Env.render()




class Agent:
    def __init__(self, env_):
        self.env = env_
        self.states_num = env_.observation_space.n
        self.action_num = env_.action_space.n
        self.Qs = np.zeros([self.states_num, self.action_num])
        #---------------------------------------
        self.now_state = env_.reset()
        self.now_reward = 0
        self.Dying = False
        self.info = None
        self.Actions = {}

    def CreateAction(self, kwargs={ 'LEFT' : 0, 'DOWN' : 1, 'RIGHT' : 2, 'UP' : 3 }):
        self.Actions.update(kwargs)

    def DoAction(self, act):
        self.now_state, self.now_reward, self.Dying, self.info = self.env.step(self.Actions[act])
        self.PrintInfo()

    def Reset(self):
        self.Dying = False
        self.now_state = self.env.reset()

    def PrintInfo(self):
        print("State: ", self.now_state, "Reward: ", self.now_reward, "Info: ", self.info)
        #print(self.Qs)

    # Train Mode

    def rargmax(self, vector):
        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]
        return random.choice(indices)

    def SelfTrain(self, count):
        Rewards_list = []

        for i in range(count):
            reward_All = 0
            self.Reset()
            while not self.Dying:

                t_state = self.now_state
                choose_action = self.rargmax(self.Qs[t_state, :])
                self.DoAction(list(self.Actions.keys())[choose_action])
                self.Qs[t_state, choose_action] = self.now_reward + np.max(self.Qs[self.now_state, :])

                reward_All += self.now_reward

            Rewards_list.append(reward_All)

        print("Final Q-Table Values\n LEFT DOWN RIGHT UP")
        print(self.Qs)
        print("Sucess rate: " + str(sum(Rewards_list)/count))



Reinforce = Game('Reinforce Man', 300, 200)
Reinforce.Register('FrozenLake-v3', 'gym.envs.toy_text:FrozenLakeEnv', {'map_name':'4x4', 'is_slippery':False})


minsu = Agent(Reinforce.GetEnv())
minsu.CreateAction()


#------------------------------------------------
def ActionAndRender(act_str):
    minsu.DoAction(act_str)
    if minsu.Dying:
        print('He\'s dead.')
        minsu.Reset()
    Reinforce.Render()
#------------------------------------------------
Reinforce.CreateCommand({
    ord('a') : [ ActionAndRender, 'LEFT' ],
    ord('d') : [ ActionAndRender, 'RIGHT' ],
    ord('w') : [ ActionAndRender, 'UP' ],
    ord('s') : [ ActionAndRender, 'DOWN' ],
})
#------------------------------------------------

minsu.SelfTrain(2000)

Reinforce.Render()

while True:

    Reinforce.Loop_Events()   # If you're simsim Just Do it


