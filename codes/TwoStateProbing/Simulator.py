from process import MP
from mdp_sampler import MDP_sampler
from alg_sampler import alg_sampler
from Clock import discrete_clock
from def_state import alg_state
from def_state import mdp_state

import scipy.stats as dist

class Simulator:

    def __init__(self, time_horizon=8) -> None:
        self.process = MP(states=[0,1], transitions=[[0.7, 0.3], [0.3, 0.7]])
        self.clock = discrete_clock()
        self.time_horizon = time_horizon
        
        self.mdp_sampler = MDP_sampler(process=self.process,timelimit=5, sampling_cost=0.5, f=[[0,1], [1,0]])
        self.mdp_sampler.get_policy()
        # self.mdp_sampler.state = mdp_state(0, 0)
        
        self.alg_sampler = alg_sampler(process=self.process, sampling_cost=0.5, f=[[0,1], [1,0]])
        # self.alg_sampler.state = alg_state(0, 0, 0)
        
        self.bernoulli_sampler = dist.bernoulli(0.5)
        
        self.mdp_values = []
        self.alg_values = []
        self.bernoulli_values = []
        
        self.actual_values = []
        
        
    
    def simulate(self):
        
        actual_value = 0
        
        while True:
            self.clock.increment()
            curr_time = self.clock.get_time()
            
            if curr_time > 10:
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
            
            if self.bernoulli_sampler.rvs():
                self.bernoulli_values.append(actual_value)
                
            else:
                self.bernoulli_values.append(0)
            
            self.mdp_sampler.update_state(mdp_action, actual_value)
            self.alg_sampler.update_state(alg_action, actual_value)
            
            self.actual_values.append(actual_value)
            
