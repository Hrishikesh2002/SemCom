import random
import numpy as np
from MP import MP

"""The states of the MDP are denoted as:  (w, w_hat, E_b).

  where w = sampled value from the process
        w_hat = last transmitted value
        E_b = The amount of energy in the battery
        


The action space (for each state) is (0,1) where 0 denotes
the sample is not transmitted and 1 denoted it is transmitted

Minimum amount of energy in the battery is 0 and maximum is E_b_max.
It takes 1 unit of energy to transmit a sample.

The reward for states is defined as:
    R(w, w_hat, E_b) = -P(w,w_hat)

where P is the similarity matrix

The value function is [no_of_w] * [no_of_w_hat] * [no_of_E_b] * [no_of_actions]
The policy is [no_of_w] * [no_of_w_hat] * [no_of_E_b] * [no_of_actions] 
The transition matrix is [no_of_w] * [no_of_w_hat] * [no_of_E_b] * [no_of_actions] * [no_of_w] * [no_of_w_hat] * [no_of E_b]





"""

class Decider:
    def __init__(self, states:list, transitions:list, actions:list, discount_factor:int, reward:list, process:MP, P:list):
        self.states = states 
        self.num_states = len(self.states)
        self.transitions = transitions
        self.discount_factor = discount_factor
        self.reward = reward
        self.actions = actions
        self.process = process
        self.last_transmitted_value = 0       # Last predicted value
        self.P = P
        
        
    # def get_value(self, state, value):
    #     return value[state[0]][state[1]][state[2]]
    
    # def get_action(self, state, policy):
    #     return policy[state[0]][state[1]][state[2]]
        
    
    def sample_next_action(self, state, policy):
        return random.choice(self.actions, weights=policy[state[0]][state[1]][state[2]])
    
    
    
    def sample_next_state(self, state, action):
        if action:
            E_b_next = state[2] - 1
            w_next = self.process.sample_next_state(state[0])
            w_hat_next = self.last_transmitted_value 
            self.last_transmitted_value = w_next
            
                  
  
        else:
            E_b_next = state[2]
            w_next = state[0]
            w_hat_next = self.last_transmitted_value
            
        return (w_next, w_hat_next, E_b_next)
    
    
            
            


def states_generator(w_max, w_hat_max, E_b_max):
    states = []
    for i in range(w_max):
        for j in range(E_b_max):
            for k in range(w_hat_max):
                states.append((i,j,k))
                
    return states



