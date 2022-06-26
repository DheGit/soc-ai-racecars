from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np

class PlantEnv(Env):
    AGE_INCR = 0.01
    CONDITION_DECAY_FACTOR = 0.9

    REPAIR_COST = 10
    MAINTENANCE_COST = 1
    
    MAINTENANCE_CONDITION_INCR = 1

    ACTION_MAINTAIN = 1
    ACTION_NO_MAINTAIN = 0

    RETIRE_AGE = 100
    DEFAULT_RECORDS = {
        "repairs": 0,
        "maintenance": 0
    }

    def __init__(self):
        self.action_space = Discrete(2) # 0: don't maintain, 1:do maintain

        self.observation_shape = (2)
        self.observation_space_ranges = ((0,100),(0,100))
        self.observation_space = Box(low=np.zeros(self.observation_shape), high=np.ones(self.observation_shape)*100)

        # STATE = (AGE, CONDITION)
        self.state = [0,100]

        self.episode_length = 10000

        self.age_incr = PlantEnv.RETIRE_AGE/self.episode_length

        self.records = dict(PlantEnv.DEFAULT_RECORDS)

    def step(self, action):
        reward = 0

        if action == PlantEnv.ACTION_MAINTAIN:
            self.runMaintenance()
            reward-=PlantEnv.MAINTENANCE_COST
            # print(f"\tMaintenance applied at age {self.state[0]} and condition {self.state[1]}")

        self.ageAndDecay()

        if self.isPlantDown():
            self.records["repairs"] += 1
            reward-=PlantEnv.REPAIR_COST
            # print(f"\tPlant Broke Down at age {self.state[0]}")

        done = (self._getAge() >= PlantEnv.RETIRE_AGE)

        info = {}

        return self.state, reward, done, info   

    def runMaintenance(self):
        self.records["maintenance"] += 1
        self._setCondition(self._getCondition() + PlantEnv.MAINTENANCE_CONDITION_INCR)

    def ageAndDecay(self):
        self._setCondition(PlantEnv.CONDITION_DECAY_FACTOR * self._getCondition())
        self._setAge(self._getAge()+self.age_incr)

    def isPlantDown(self):
        chance = random.random()

        return chance > (1-(self._getCondition()/100))*(self._getAge()/self.RETIRE_AGE)

    def reset(self):
        # STATE = (AGE, CONDITION)
        self.state = [0,100]

        return self.state

    def render(self):
        # karo kabhi isko bhi implement
        pass

    def _setState(self, age, condition):
        self.state = [age,condition]

    def _getAge(self):
        return self.state[0]
    def _setAge(self, new_age):
        self.state[0]=new_age
    def _getCondition(self):
        return self.state[1]
    def _setCondition(self, new_condition):
        self.state[1]=new_condition

    def getNumActions(self):
        return self.action_space.n

    def getStateRanges(self):
        return self.observation_space_ranges 

    def resetRecords(self):
        self.records = dict(PlantEnv.DEFAULT_RECORDS)

    def getRecords(self):
        return self.records

global_env = PlantEnv()

def test__gettersAndSetters():
    global global_env

    print(f"Age of plant: {global_env._getAge()}")
    print(f"Condition of plant: {global_env._getCondition()}")

    global_env._setAge(69)
    global_env._setCondition(92)

    print(f"Age of plant: {global_env._getAge()}")
    print(f"Condition of plant: {global_env._getCondition()}")