from cmath import inf
from plant_model import PlantEnv
import numpy as np

def getProduct(t):
    prod = 1
    try:
        for a in t:
            prod *= a
    except:
        return inf

    return prod

def isIterable(v):
    try:
        itr = iter(v)
    except TypeError:
        return False

    return True
    

class PlantMaintainer:
    def __init__(self, state_shape, num_actions, alpha = 0.1, gamma = 0.5, state_convertor = None):
        self.state_shape = state_shape
        self.tot_actions = num_actions

        # self.q = np.random.rand(num_states,num_actions)
        q_shape = (state_shape)+(num_actions,)
        self.q = np.random.rand(getProduct(q_shape)).reshape(q_shape)
        self.lr = alpha
        self.disc = gamma
        
        self.state_convertor = state_convertor

    def improveQ(self, state, action, reward, new_state):
        if not self.state_convertor is None:
            state = self.state_convertor.getDiscreteState(state)
            new_state = self.state_convertor.getDiscreteState(new_state)

        
        cur_val = self.q[state][action]
        # self.q[state][action] = (1-self.lr)*cur_val + self.lr*(reward + self.disc*max(self.q[new_state]))
        self._setQ(state, action, (1-self.lr)*cur_val + self.lr*(reward + self.disc*max(self._getActionValues(new_state))))

    def decideAction(self,state):
        if not self.state_convertor is None:
            state = self.state_convertor.getDiscreteState(state)

        res = 0
        action_values = self._getActionValues(state)

        max_q = action_values[res]

        for i in range(1,self.tot_actions):
            this_q = action_values[i]
            if this_q > max_q:
                max_q = this_q
                res = i

        return res

    def _setQ(self, state, action, q):
        if not isIterable(state):
            state = (state,)
        
        target = self.q

        for ind in state:
            target = target[ind]

        target[action] = q

    def _getQ(self, state, action):
        if not isIterable(state):
            state = (state,)

        target = self.q

        for ind in state+(action,):
            target = target[ind]

        return target

    def _getActionValues(self, state):
        if not isIterable(state):
            state = (state,)

        target = self.q

        for ind in state:
            target = target[ind]

        return target


'''
converts a continuous state to a discrete one (from 0 to num_states-1)
'''
class StateConvertor:
    def __init__(self, state_ranges, num_segments):
        if not isIterable(state_ranges):
            state_ranges = (state_ranges,)
        if not isIterable(num_segments):
            num_segments = (num_segments,)

        if len(state_ranges) != len(num_segments):
            print("The length of state_ranges and num_segments must be the same")
            return
        for state_range in state_ranges:
            if state_range[0] >= state_range[1]:
                print("Bad state range, failed initialisation of StateConvertor")
                return
        for num_segment in num_segments:
            if num_segment <= 0:
                print("Number of segments must be posittive, failed initialisation of StateConvertor")
                return
                
        self.range = state_ranges
        self.num_segments = num_segments
    
    def getDiscreteState(self, cont_state):
        if not isIterable(cont_state):
            cont_state = (cont_state,)

        res = []

        for i in range(len(self.num_segments)):
            frac = (cont_state[i]-self.range[i][0])/(self.range[i][1]-self.range[i][0])
            this_res = int(self.num_segments[i]*frac)

            if this_res >= self.num_segments[i]:
                res.append(self.num_segments[i]-1)
            else:
                res.append(this_res)

        return tuple(res)

def main(ep=100):
    env = PlantEnv()
    resolution = (100,100)
    state_ranges = env.getStateRanges()
    bot = PlantMaintainer((100,100), env.getNumActions(), state_convertor = StateConvertor(state_ranges,resolution))

    episodes = ep

    for ep in range(0,episodes):
        runEpisode(env, bot,index=ep)

def runEpisode(env, agent, index = -1):
    episode_result = ""

    if index >= 0:
        episode_result = "Episode "+str(index)+": Scored {score}"
    else:
        episode_result = "This episode: Scored {score}"

    state = env.reset()
    done = False
    score = 0

    while not done:
        # action = env.action_space.sample()
        action = agent.decideAction(state)
        new_state, reward, done, info = env.step(action)
        score += reward
        
        agent.improveQ(state, action, reward, new_state)

        state = new_state

        # if env._getAge() < 90:
        #     print(f"age: {env._getAge()}")

    print(episode_result.format(score=score))
    print(f"Episode Statistics: {env.getRecords()}")
    env.resetRecords()

# main()