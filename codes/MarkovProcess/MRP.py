import random
from venv import create

class MarkovRewardProcess:
    
    # value_function = {}
    
    def __init__(self, states, transitions, reward):
        self.states = states
        self.num_states = len(states)
        self.transitions = transitions
        self.reward = reward
        self.value_function = {}
        
        self.create_value_dict()
        
    def create_value_dict(self):
        for state in self.states:
            self.value_function[state] = 0
        
    def __repr__(self) -> str:
        return f"MarkovRewardProcess(states={self.states}, transitions={self.transitions}, reward={self.reward})"
    
    def sample_next_state(self, state: str) -> str:
        return random.choices(list(self.transitions[state].keys()), weights=list(self.transitions[state].values()))[0]
        
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
        while state != 'TERMINAL' and len(path) < path_length:
            state = self.sample_next_state(state)
            path.append(state)
            sum_reward += self.get_reward(state)
        return sum_reward, path
    
    
    def inverse_solver(self):
        self.num_states
    
    def solver(self):
        num_states = self.num_states
        
        states_mat = range(num_states)
        transition_mat = [[0 for i in range(num_states)] for j in range(num_states)]
        reward_mat = [0 for i in range(num_states)]
        
        
        
    

    

def driver():
    states = ['A', 'B', 'C', 'D', 'E', 'TERMINAL']
    transitions = {'A':{ 'A':0.2, 'B':0.3, 'C':0.4, 'TERMINAL':0.1}, 'B':{ 'A':0.4, 'B':0.1, 'C':0.5}, 'C':{ 'A':0.3, 'B':0.4, 'C':0.1, 'TERMINAL':0.2}, 'D':{ 'D':0.1, 'E':0.9}, 'E':{ 'D':0.9, 'E':0.1}, 'TERMINAL':{ 'TERMINAL':1.0}}
    reward = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'TERMINAL':1}
    
    
    markovRewardProcess = MarkovRewardProcess(states, transitions, reward)
    
    sum_reward, path = markovRewardProcess.sample_path_reward('A', 20)
    
    print(f"Total reward={sum_reward} \npath={path}")
    

if __name__ == "__main__":
    driver()
    
    