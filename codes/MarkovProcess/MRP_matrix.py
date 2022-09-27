import random
import numpy as np
import sys


class MarkovRewardProcess:
    
    def __init__(self, states, transitions, reward, discount_factor=0.3) -> None:
        self.states = states
        self.num_states = len(self.states)
        self.transitions = transitions
        self.reward = reward
        self.discount_factor = discount_factor
        
        
    def __repr__(self) -> str:
        return f"MarkovRewardProcess(states={self.states}, transitions={self.transitions}, reward={self.reward})"
    
    def sample_next_state(self, state: str) -> str:
        return random.choices(self.states, weights=list(self.transitions[state]))[0]
        
    def sample_path(self, state, path_length) -> list:
        path = [state]
        while len(path) < path_length:
            state = self.sample_next_state(state)
            path.append(state)
        return path
    
    def get_reward(self, state):
        return self.reward[state]
    
    def sample_path_reward(self, state, path_length) -> list:
        path = [state]
        sum_reward = 0
        i = 1
        while len(path) < path_length:
            state = self.sample_next_state(state)
            path.append(state)
            sum_reward += self.get_reward(state) * (self.discount_factor ** i)
            i += 1
        return sum_reward, path
    
    
    def inverse_solver(self):
        id = np.identity(self.num_states)
        
        states_mat = np.array(range(self.num_states))
        transition_mat = np.array(self.transitions)
        
        value_vec = np.matmul(np.linalg.inv(id - self.discount_factor * transition_mat) , np.array(self.reward))
        
        print(f"\nValue function = {value_vec}")
        
        
        
    
    def policy_evaluation(self):
        value_vec = np.array([0 for i in range(self.num_states)])
        diff = sys.maxsize
        while(diff > 0.0001):
            prev_value_vec = value_vec.copy()
            value_vec = self.reward + self.discount_factor * np.array([np.dot(value_vec, self.transitions[i]) for i in range(self.num_states)])
            diff = np.linalg.norm(value_vec - prev_value_vec)
        
        print(f"\nValue function = {value_vec}")
        
        
    

def driver():
    states = [0,1,2,3]
    transitions = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1]]
    reward = [0, 0, 0, 1]
    
    markovRewardProcess = MarkovRewardProcess(states, transitions, reward)
    
    # sum_reward, path = markovRewardProcess.sample_path_reward(0, 20)
    
    # print(f"Total reward={sum_reward} \npath={path}")
    
    markovRewardProcess.inverse_solver()
    markovRewardProcess.policy_evaluation()
    
if __name__ == "__main__":
    driver()
    