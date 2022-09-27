import random
import numpy as np
from MP import MP
from scipy.stats import bernoulli

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


The battery is has maximum charge E_b_max and minimum charge E_b_min. It charges at each transition according to a bernoulli rv with probability p.





"""

class Battery:
    def __init__(self, E_b_max, E_b, E_b_min=0,  p=0.5) -> None:
            
        self.E_b_max = E_b_max
        self.E_b_min = E_b_min
        self.E_b = E_b
        self.p = p
        self.rv = bernoulli(p)
        
    def __repr__(self) -> str:
        return f"Battery: charging: {self.E_b} p: {self.p} max_charge: {self.E_b_max} min_charge: {self.E_b_min} "
    
    def discharge(self):
        self.E_b -= 1
        
    def charge(self):
        if self.rv.rvs() == 1 and self.E_b < self.E_b_max:
            self.E_b += 1
            

class Decider:
    def __init__(self, states:list, transitions:list, actions:list, discount_factor:int, process:MP, P:list, bat:Battery):
        self.states = states 
        self.num_states = (len(self.states), len(self.states[0]), len(self.states[0][0]) )
        self.transitions = transitions
        self.discount_factor = discount_factor
        self.actions = actions
        self.process = process
        self.last_transmitted_value = 0       # Last predicted value
        self.P = P
        self.bat = bat
        
        self.define_reward()
        
        
    def define_reward(self):
        reward = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2], len(self.actions)))
        for i1 in range(self.num_states[0]):
            for i2 in range(self.num_states[1]):
                for i3 in range(self.num_states[2]):
                    for j in range(len(self.actions)):
                        reward[i1][i2][i3][j] = -self.P[self.states[i1][i2][i3]][self.states[i1][i2][i3]]
                        
        self.reward = reward
        
    
    def sample_next_action(self, state, policy):
        return random.choice(self.actions, weights=policy[state[0]][state[1]][state[2]])
    
    
    
    def sample_next_state(self, state, action):
        self.bat.charge()
        
        if action:
            self.bat.discharge()
            E_b_next = self.bat.E_b
            w_next = self.process.sample_next_state(state[0])
            w_hat_next = self.last_transmitted_value 
            self.last_transmitted_value = w_next
            
                  
  
        else:
            E_b_next = state[2]
            w_next = state[0]
            w_hat_next = self.last_transmitted_value
            
        return (w_next, w_hat_next, E_b_next)
    
    def policy_iteration(self):
        #initialize policy
        policy = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2], len(self.actions)))
        for i in range(self.states[0]):
            for j in range(self.states[1]):
                for k in range(self.states[2]):
                    policy[i][j][k] = np.ones(len(self.actions))/len(self.actions)
                    
        #initialize value function
        value = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2]))
        
        #initialize policy evaluation
        new_value = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2]))
        
        #initialize policy improvement
        new_policy = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2], len(self.actions)))
        
        #initialize policy stable
        policy_stable = False
        
        while not policy_stable:
            #policy evaluation
            while True:
                new_value = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2]))
                for i1 in range(self.num_states[0]):
                    for i2 in range(self.num_states[1]):
                        for i3 in range(self.num_states[2]):
                            for j in range(len(self.actions)):
                                pi = policy[i1][i2][i3][j]
                                environmental_expectation = np.dot(self.transitions[i1][i2][i3][j], value)
                                new_value[i] += pi * (self.reward[i1][i2][i3][j] + self.discount_factor * environmental_expectation)
                                
                                
                                
                if np.allclose(new_value, value, atol=1e-10):
                    break
                
                value = new_value
            
            #policy improvement
            
            for i1 in range(self.num_states[0]):
                for i2 in range(self.num_states[1]):
                    for i3 in range(self.num_states[2]):
                        max_value = -np.inf
                        action_reward_list = []
                        for j in range(len(self.actions)):
                            state_action_value = self.reward[i1][i2][i3][j] + self.discount_factor * np.dot(self.transitions[i1][i2][i3][j], value)
                            action_reward_list.append(state_action_value)
                            if(state_action_value > max_value):
                                max_value = state_action_value
                            
                        new_policy[i1][i2][i3] = np.zeros(len(self.actions))
                        num_max_actions = action_reward_list.count(max_value)
                        
                        for k in range(len(self.actions)):
                            if(action_reward_list[k] == max_value):
                                new_policy[i1][i2][i3][k] = 1/num_max_actions
                                
            #check if policy is stable
            if np.allclose(policy, new_policy, atol=1e-10):
                policy_stable = True
            
            policy = new_policy
            
        return policy, value
    




def states_generator(w_max, w_hat_max, E_b_max):
    states = []
    for i in range(w_max):
        for j in range(E_b_max):
            for k in range(w_hat_max):
                states.append((i,j,k))
                
    return states



