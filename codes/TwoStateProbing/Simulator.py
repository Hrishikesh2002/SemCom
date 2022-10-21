from process import MP
from mdp_sampler import MDP_sampler
from alg_sampler import alg_sampler
from Clock import discrete_clock
import numpy as np

import scipy.stats as dist

class Simulator:

    def __init__(self, time_horizon=30) -> None:
        
        
        self.process = MP(states=[0,1], transitions=[[0.7, 0.3], [0.3, 0.7]])
        self.clock = discrete_clock()
        self.time_horizon = time_horizon
        
        self.mdp_sampler = MDP_sampler(process=self.process,timelimit=5, sampling_cost=0.5, f=[[0,1], [1,0]])
        self.mdp_sampler.get_policy()
        # self.mdp_sampler.state = mdp_state(0, 0)
        
        self.alg_sampler = alg_sampler(process=self.process, sampling_cost=0.5, f=[[0,1], [1,0]])
        # self.alg_sampler.state = alg_state(0, 0, 0)
        
        self.bernoulli_sampler = dist.bernoulli(0.5)
        
        self.alg_error = 0
        self.mdp_error = 0
        self.bernoulli_error = 0
        
        self.mdp_values = []
        self.alg_values = []
        self.bernoulli_values = []
        
        self.actual_values = []
        
    def error_expectation(self, last_sampled_value, time_difference):  # This is the E[f(s(t), s_hat(t))] function
        transitions_new = np.linalg.matrix_power(self.process.transitions, time_difference)
        expectation =  np.dot(transitions_new[last_sampled_value] , self.mdp_sampler.f[self.mdp_sampler.mle])
        return expectation
        

    def updateError(self, mdp_action, alg_action, bernoulli_action):
        if mdp_action:
            self.mdp_error += self.mdp_sampler.sampling_cost
            
        else:
            self.mdp_error += self.error_expectation(self.mdp_sampler.mle, self.mdp_sampler.state.last_sample_time)
            
        if alg_action:
            self.alg_error += self.alg_sampler.sampling_cost
            
        else:
            self.alg_error += self.error_expectation(self.alg_sampler.mle, self.alg_sampler.state.last_sample_time)
            
        if bernoulli_action:
            self.bernoulli_error += 0.5
        
        else:
            self.bernoulli_error += self.error_expectation(0, 1)
        
        
        
        
        
    
    def simulate(self):
        
        
        actual_value = 0
        
        while True:
            self.clock.increment()
            curr_time = self.clock.get_time()
            
            if curr_time > self.time_horizon:
                break
            
            
            actual_value = self.process.sample_next_state(actual_value)
            
            
            mdp_action = self.mdp_sampler.get_current_action()
            alg_action = self.alg_sampler.get_current_action()
            
            if mdp_action:
                self.mdp_values.append(actual_value)
                
            else:
                self.mdp_values.append(self.mdp_sampler.mle)
            
            if alg_action:
                self.alg_values.append(actual_value)
                
            else:
                self.alg_values.append(self.alg_sampler.mle)
                
            bernoulli_action = self.bernoulli_sampler.rvs()
            
            if bernoulli_action:
                self.bernoulli_values.append(actual_value)
                
            else:
                self.bernoulli_values.append(0)
            
            self.mdp_sampler.update_state(mdp_action, actual_value)
            self.alg_sampler.update_state(alg_action, actual_value)
            
            self.actual_values.append(actual_value)
            
            self.updateError(mdp_action, alg_action, bernoulli_action)
            
    # choose one of the samplers and return the error after convergence
    # def simulate_one(self, sampler):
        
    #     if sampler == "mdp":
    #         sampler = self.mdp_sampler
            
    #     if sampler == "alg":
    #         sampler = self.alg_sampler
            
    #     if sampler == "bernoulli":
    #         sampler = self.bernoulli_sampler
        
            
    #     actual_value = 0
        
    #     while True:
    #         self.clock.increment()
    #         curr_time = self.clock.get_time()
            
    #         if curr_time > self.time_horizon:
    #             break
            
            
    #         actual_value = self.process.sample_next_state(actual_value)
            
            
    #         action = sampler.get_current_action()
            
    #         if action:
    #             self.mdp_values.append(actual_value)
                
    #         else:
    #             self.mdp_values.append(sampler.mle)
                
        
            
    #         self.sampler.update_state(action, actual_value)
            
    #         self.actual_values.append(actual_value)
            
    #         self.updateError(action)
            
            
