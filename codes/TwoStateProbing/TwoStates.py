from concurrent.futures import process
import random
import numpy as np

class MP:
    def __init__(self, states, transitions) -> None:
        self.states = states
        self.num_states = len(states)
        self.transitions = np.array(transitions)
        
    
    def __repr__(self) -> str:
        return f"MarkovProcess(states={self.states}, transitions={self.transitions})"
    
    def sample_next_state(self, state:int) -> int:
        return random.choices(self.states, weights=self.transitions[state])[0]
    
    def sample_path(self, state, path_length) -> list:
        path = [state]
        while len(path) < path_length:
            state = self.sample_next_state(state)
            path.append(state)
        
        return path
    


class MLE:
    
    def __init__(self, states, transitions) -> None:
        self.states = states
        self.transitions = transitions



class Decider:
    def __init__(self, process, timelimit=5, sampling_cost=1, f=[[0,1], [1,0]]) -> None:
       self.process = process
       self.timelimit = timelimit
       self.actions = [0,1]
       self.f = f # predicted_value * actual_value
       self.sampling_cost = sampling_cost
       
       
    def mle(self, time_difference, last_state):
        
        transitions_static = np.power(self.transitions, time_difference)
        return transitions_static.index(np.max(transitions_static[last_state]))
     
     
     
    def define_states(self):
        # Define states as (last sampled state, time since last sample) i.e. (s,t)
        self.states = [(i,j+1) for i in range(len(self.process.states)) for j in range(self.timelimit)]
        
        
         
    def define_transitions(self):
        
        self.transitions = {current_state: {action: {next_state: 0 for next_state in self.states} for action in self.actions} for current_state in self.states}
        
        for current_state in self.states:
            transitions_new = np.linalg.matrix_power(self.process.transitions, current_state[1])
            
            for action in self.actions:
                for next_state in self.states:
                    if action and next_state[1] == 0:
                        self.transitions[current_state][action][next_state] = transitions_new[current_state[0]][next_state[0]]
                        
                    elif not action and next_state[1] == current_state[1] + 1:
                        self.transitions[current_state][action][next_state] = transitions_new[current_state[0]][next_state[0]]
                        
                    else:
                        self.transitions[current_state][action][next_state] = 0
                        
                        
                        
    def error_expectation(self, predicted_state, time_difference):  # This is the E[f(s(t), s_hat(t))] function
        transitions_new = np.linalg.matrix_power(self.process.transitions, time_difference)
        expectation = transitions_new[predicted_state][0] * self.f[predicted_state][0] + transitions_new[predicted_state][1] * self.f[predicted_state][0]
        return expectation

                        
                        
    def define_reward(self):
        for current_state in self.states:
            for action in self.actions:
                if action:
                    self.reward[current_state][action] = -1 * self.sampling_cost
                    
                else:
                    self.reward[current_state][action] = -1 * self.error_expectation(current_state[0], current_state[1])
                    
    
    def policy_iteration(self):
        # Initialize policy
        new_policy = np.zeros((len(self.states), len(self.actions)))
        
   
    
    
        
        





class Simulator:
    pass
    
    
    
    
    