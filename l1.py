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
        
    def update(self, action):
        if (action == 0):
            if self.plantFailure(): # Repairs required
                self.age = 0
                self.p_fail = 0
                return (self.determineReward(failed=True), self.age)
            else: # Phew
                self.age += 1
                self.p_fail += 0.05
                return (self.determineReward(), self.age)
        
        if (action == 1):
            self.reset()
            
            return (self.determineReward(maintenance=True), self.age)

    def plantFailure(self):
        return np.random.random() < self.p_fail

    def determineReward(self, failed=False, maintenance=False):
        if failed:
            return PlantEnv.BASE_REWARD-PlantEnv.REPAIR_COST

        if maintenance:
            return PlantEnv.BASE_REWARD - PlantEnv.MAINTENANCE_COST
        
        return PlantEnv.BASE_REWARD

    def reset(self):
        self.p_fail = 0
        self.age = 0

AGE_RANGE = 100
NUM_ACTIONS = 2
Q = np.zeros(shape = (AGE_RANGE, NUM_ACTIONS))   

plant = PlantEnv()
cur_age = 0

ACTION_LIST = [0,1]

# going through ITERATIONS epochs
for i in range(EPOCHS):
    if (np.random.random() < p_explore): #Explore
        action = random.choice(ACTION_LIST) 
    else: #Exploit
        action = np.argmax(Q[cur_age])
    
    reward, next_age = plant.update(action)
    
    Q[cur_age][action] = (1 - LEARNING_RATE)*Q[cur_age][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q[next_age]))    
    
    cur_age = next_age
    p_explore -= P_EXPLORE_DECR


plant.reset()
cur_age = 0

for i in range(10000):
    action = np.argmax(Q[cur_age])
    reward, cur_age = plant.update(action)

# Plotting
plt.plot(Q[0:100, 0], label="No Maintenance")
plt.plot(Q[0:100, 1], label = "Yes Maintenance")
plt.xlabel("Age")
plt.ylabel("Q-Value")
plt.legend()
plt.show()