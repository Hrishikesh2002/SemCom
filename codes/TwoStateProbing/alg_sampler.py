from def_state import alg_state
from Clock import discrete_clock
import numpy as np


class alg_sampler:
    
    def __init__(self, process, sampling_cost, state=alg_state(0, 0, 0), f = [[0,1],[1,0]]) -> None:
        self.process = process
        self.sampled_state = 0
        self.sampling_cost = sampling_cost
        self.state = state
        self.f = f
        self.calc_mle()
        
    
    def calc_mle(self):
        T_final = T_new = self.process.transitions
        
        while True:
            T_new = np.dot(T_new, self.process.transitions)
            if np.allclose(T_final, T_new, atol=1e-20, rtol=0):
                break
            
            T_final = T_new
            
        mle = np.argmax(T_final[0])
        
        self.mle = mle
    
    
    
    
    def error_expectation(self, last_sampled_value, time_difference):  # This is the E[f(s(t), s_hat(t))] function
        transitions_new = np.linalg.matrix_power(self.process.transitions, time_difference)
        expectation =  np.dot(transitions_new[last_sampled_value] , self.f[self.mle])
        return expectation


    
    def calc_next_cost(self, curr_state:alg_state):
        next_state_cost = curr_state.curr_state_cost + self.error_expectation(curr_state.last_sampled_state, curr_state.last_sample_time + 1)
        
        return next_state_cost
        
        
        
    def update_state(self, curr_action:int, sampled_state:int):
        
        if curr_action:
            self.state = alg_state(sampled_state, 0, 0)
        
        else:
            self.state = alg_state(self.state.last_sampled_state, self.state.last_sample_time + 1, self.calc_next_cost(self.state))
        
        
    def get_current_action(self):
        if(self.state.curr_state_cost > self.sampling_cost):
            return 1
        else:
            return 0
        