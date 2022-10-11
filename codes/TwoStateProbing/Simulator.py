from mdp_sampler import MDP_sampler
from process import MP




class Simulator:
    
    
    def __init__(self) -> None:
        self.process = MP(states=[0,1], transitions=[[0.7, 0.3], [0.3, 0.7]])
        self.mdp_sampler = MDP_sampler(process=self.process, timelimit=5, sampling_cost=0.5, f=[[0,1], [1,0]])
        # self.bernoulli_sampler = dist.bernoulli(0.5)
        
    def get_policy(self):
        self.mdp_sampler.solve()
        
def driver():
    
    simulator = Simulator()
    # print(simulator.MDP_sampler.states)
    simulator.get_policy()
    
driver()
        
        
        
        
    
        

        
        
    
    
    
    
    
    