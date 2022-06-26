import random
from gym import Env
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.1

EPOCHS = 1000

p_explore = 0.8
P_EXPLORE_DECR = p_explore / (EPOCHS//2) 

'''
Class enclosing plant behaviour
'''
class PlantEnv(Env):
    REPAIR_COST = 120
    MAINTENANCE_COST = 30
    BASE_REWARD = 100

    def __init__(self):
        self.p_fail = 0
        self.age = 0
        self.condition = 1
        
    def update(self, action):
        if (action == 0):
            if self.plantFailure(): # Repairs required
                self.age = 0
                self.p_fail = 0
                return (self.determineReward(failed=True), self.age, self.condition)
            else: # Phew
                self.age += 1
                self.p_fail += 0.05
                self.condition -= 0.1*random.random()
                return (self.determineReward(), self.age, self.condition)
        
        if (action == 1):
            self.reset()
            
            return (self.determineReward(maintenance=True), self.age, self.condition)

    def plantFailure(self):
        return max(0.2+0.05*self.age, 1-self.condition)

    def determineReward(self, failed=False, maintenance=False):
        if failed:
            return PlantEnv.BASE_REWARD-PlantEnv.REPAIR_COST

        if maintenance:
            return PlantEnv.BASE_REWARD - PlantEnv.MAINTENANCE_COST
        
        return PlantEnv.BASE_REWARD

    def reset(self):
        self.p_fail = 0
        self.age = 0
        self.condition = 1


AGE_RANGE = 100
CONDITION_RANGE = 50
NUM_ACTIONS = 2
Q = np.zeros(shape = (AGE_RANGE, CONDITION_RANGE, NUM_ACTIONS))   
plant = PlantEnv()
cur_age = 0
cur_condition = CONDITION_RANGE - 1

ACTION_LIST = [0,1]

rewards_record = [0]

# going through ITERATIONS epochs
for i in range(EPOCHS):
    if (np.random.random() < p_explore): #Explore
        action = random.choice(ACTION_LIST) 
    else: #Exploit
        action = np.argmax(Q[cur_age][cur_condition])
    
    reward, next_age, next_condition = plant.update(action)
    next_condition = int(CONDITION_RANGE*next_condition)
    if next_condition >= CONDITION_RANGE:
        next_condition=CONDITION_RANGE-1
    
    Q[cur_age][cur_condition][action] = (1 - LEARNING_RATE)*Q[cur_age][cur_condition][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q[next_age][next_condition]))    
    
    cur_age = next_age
    cur_condition = next_condition
    p_explore -= P_EXPLORE_DECR

    rewards_record.append((rewards_record[-1]*len(rewards_record)+reward)/(len(rewards_record)+1))


plant.reset()
cur_age = 0

for i in range(10000):
    action = np.argmax(Q[cur_age][cur_condition])
    reward, cur_age, cur_condition = plant.update(action)

# Plotting
plt.plot(rewards_record, label="Average Reward")
plt.xlabel("Time")
plt.ylabel("Average Rewards")
plt.show()
