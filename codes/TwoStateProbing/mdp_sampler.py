import mdptoolbox
import numpy as np
import math
from def_state import mdp_state

class MDP_sampler:
    def __init__(self, process, timelimit=5, sampling_cost=1, f=[[0,1], [1,0]]) -> None:
        self.state = mdp_state(0,0)
        self.process = process
        self.timelimit = timelimit
        self.actions = [0,1]
        self.f = f # predicted_value * actual_value
        self.sampling_cost = sampling_cost
        self.define_states()
        self.define_transitions()
        self.calc_mle()
        self.define_reward()
       
       
    def calc_mle(self):
        T_final = T_new = self.process.transitions
        
        while True:
            T_new = np.dot(T_new, self.process.transitions)
            if np.allclose(T_final, T_new, atol=1e-20, rtol=0):
                break
            
            T_final = T_new
            
        mle = np.argmax(T_final[0])
        
        self.mle = mle
        
        
     
     
     
    def define_states(self):
        # Define mdp_state as (last sampled state, time since last sample) i.e. (s,t)
        self.states = [mdp_state(i,j) for i in range(len(self.process.states)) for j in range(self.timelimit)]
        
        
         
    def define_transitions(self):
        
        self.transitions = np.zeros((len(self.actions), len(self.states), len(self.states)))
        
        for i, current_state in enumerate(self.states):
            transitions_new = np.linalg.matrix_power(self.process.transitions, current_state.last_sample_time)
            
            for j, action in enumerate(self.actions):
                
                for k, next_state in enumerate(self.states):
                    if action and next_state.last_sample_time == 0:
                        self.transitions[j][i][k] = round(transitions_new[current_state.last_sampled_value][next_state.last_sampled_value],4)
                        
                    elif not action and next_state.last_sample_time == current_state.last_sample_time + 1 and next_state.last_sampled_value == current_state.last_sampled_value:
                        self.transitions[j][i][k] = 1
                        
                    elif not action and current_state.last_sample_time == self.timelimit - 1 and next_state.last_sample_time == 0 and next_state.last_sampled_value == 0:
                        self.transitions[j][i][k] = 1
                    
                        
                    else:
                        self.transitions[j][i][k] = 0
                        
                        
                        
        
                        
                        
    def error_expectation(self, last_sampled_value, time_difference):  # This is the E[f(s(t), s_hat(t))] function
        transitions_new = np.linalg.matrix_power(self.process.transitions, time_difference)
        expectation =  np.dot(transitions_new[last_sampled_value] , self.f[self.mle])
        return expectation


                        
    def define_reward(self):
        self.reward = np.zeros((len(self.states), len(self.actions)))
        
        for i, current_state in enumerate(self.states):
            for j, action in enumerate(self.actions):
                if action:
                    self.reward[i][j] = -1 * self.sampling_cost
                    
                else:
                    self.reward[i][j] = -1 * self.error_expectation(current_state.last_sampled_value, current_state.last_sample_time)
                    
                    if current_state.last_sample_time == self.timelimit - 1 :
                        self.reward[i][j] = -1 * math.inf
                

# Functions needed for the simulation       

    def get_policy(self):
        
        pi = mdptoolbox.mdp.PolicyIteration(self.transitions, self.reward, 0.9)
        pi.run()
        self.policy = pi.policy
        
    
    def update_state(self, action, sampled_value=None):
        if action:
            self.state = mdp_state(sampled_value, 0)
            
        else:
            self.state = mdp_state(self.state.last_sampled_value, self.state.last_sample_time + 1)
            
        return self.state
    
    def get_current_action(self):
        return self.policy[self.state.last_sampled_value * self.timelimit + self.state.last_sample_time]
        
