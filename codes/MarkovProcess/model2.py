import random
import numpy as np
from MP import MP
from scipy.stats import bernoulli

# debugMode = False



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
		if(self.E_b == 0):
			raise Exception("Battery is empty")
		self.E_b -= 1
		
	def charge(self):
		if self.rv.rvs() == 1 and self.E_b < self.E_b_max:
			self.E_b += 1
		
class Decider:
    def __init__(self, states:list, process:MP, P:list, battery:Battery, actions:list = [0, 1], discount_factor:int=0.8):
        self.states = states
        self.process = process
        self.battery = battery
        self.actions = actions
        self.discount_factor = discount_factor
        self.P = P
        num_actions = len(self.actions)
        self.policy = {current_state: {action:1/num_actions for action in self.actions} for current_state in self.states}
        self.value = {current_state: 0 for current_state in self.states}
        self.define_transitions()
        self.define_reward()
        # self.transitions = self.define_transitions()
        # self.reward = self.define_reward()
        # self.policy = self.initiate_policy()
        
    def define_transitions(self):
        self.transitions = {current_state: {action: {next_state: 0 for next_state in self.states} for action in self.actions} for current_state in self.states}
        
        for current_state in self.states:
            for action in self.actions:
                for next_state in self.states:
                    if action:
                        if next_state[1] == next_state[0]:
                            if current_state[2] != self.battery.E_b_min:
                                if next_state[2] == current_state[2]:
                                    self.transitions[current_state][action][next_state] = self.process.transitions[current_state[0]][next_state[0]] * self.battery.p
                                elif next_state[2] == current_state[2] - 1:
                                    self.transitions[current_state][action][next_state] = self.process.transitions[current_state[0]][next_state[0]] * (1 - self.battery.p)
                                else:
                                    self.transitions[current_state][action][next_state] = 0
                                    
                            else:
                                if next_state[2] == current_state[2]:
                                    self.transitions[current_state][action][next_state] = self.process.transitions[current_state[0]][next_state[0]]
                                else:
                                    self.transitions[current_state][action][next_state] = 0
                    
                    else:
                        if current_state[1] == next_state[1]:
                            
                            if current_state[2] != self.battery.E_b_max:
                                
                                if next_state[2] == current_state[2]:
                                    self.transitions[current_state][action][next_state] = self.process.transitions[current_state[0]][next_state[0]] * (1-self.battery.p)
                                elif next_state[2] == current_state[2] + 1:
                                    self.transitions[current_state][action][next_state] = self.process.transitions[current_state[0]][next_state[0]] * self.battery.p
                                else:
                                    self.transitions[current_state][action][next_state] = 0
                                    
                            else:
                                if next_state[2] == current_state[2]:
                                    self.transitions[current_state][action][next_state] = self.process.transitions[current_state[0]][next_state[0]]
                                else:
                                    self.transitions[current_state][action][next_state] = 0
                            
                        else:
                            self.transitions[current_state][action][next_state] = 0
                            
        # print("\n\nTransitions:  ", self.transitions)
                    
            
    def define_reward(self):
        self.reward = {current_state: {action: 0 for action in self.actions} for current_state in self.states}
        
        for current_state in self.states:
            for action in self.actions:
                self.reward[current_state][action] = -self.P[current_state[0]][current_state[1]]
            
            
    def policy_iteration(self):
        # print('a')
        self.policy_stable = False
        
        while not self.policy_stable:
            self.policy_evaluation()
            self.policy_improvement()
            
    def policy_evaluation(self):
        
        while True:
            self.new_value = {current_state: 0 for current_state in self.states}
            
            #iterate over all states
            for current_state in self.states:
                for action in self.actions:
                    action_value = self.reward[current_state][action]
                    pi = self.policy[current_state][action]
                    
                    for next_state in self.states:
                        action_value +=  self.transitions[current_state][action][next_state] * self.discount_factor * self.value[next_state]
                        
                    self.new_value[current_state] += pi * action_value
                    
            if self.value_convergence():
                break
                
            self.value = self.new_value
                        
                
                        
    def policy_improvement(self):
        self.new_policy = {current_state: {action: 0 for action in self.actions} for current_state in self.states}
        
        for current_state in self.states:
            
            max_value = -np.inf
            action_reward_list = []
            
            for action in self.actions:
                action_reward = self.reward[current_state][action]
                for next_state in self.states:
                    action_reward += self.transitions[current_state][action][next_state] * self.discount_factor * self.value[next_state]
                if action_reward > max_value:
                        max_value = action_reward
                action_reward_list.append(action_reward)

            
            num_max_actions = action_reward_list.count(max_value)
            
            for action in self.actions:
                if action_reward_list[action] == max_value:
                    self.new_policy[current_state][action] = 1/num_max_actions
            
            
        if self.policy_convergence():
            self.policy_stable = True
            
        self.policy = self.new_policy
        
        
    def value_convergence(self):
        val_arr = [v for v in self.value.values()]
        new_val_arr = [v for v in self.new_value.values()]
        
        # print("\n\nval_arr: ", val_arr)
        # print("\n\nnew_val_arr: ", new_val_arr)
        
        if(np.allclose(val_arr, new_val_arr, atol=1e-4)):
            return True
        else:
            return False
        
            
    def policy_convergence(self):
        pol_arr = [[v[j] for j in [0,1]] for v in self.policy.values()]
        new_pol_arr = [[v[j] for j in [0,1]] for v in self.new_policy.values()]
        
        # print("\n\npol_arr: ", pol_arr)
        # print("\n\nnew_pol_arr: ", new_pol_arr)
        
        if(np.allclose(pol_arr, new_pol_arr, atol=1e-4)):
            return True
        
        else:
            return False
            
                
                
                


def states_generator(no_of_w, no_of_w_hat, no_of_E_b):
    # states = [(0,0,0) for i in range(no_of_w * no_of_w_hat * no_of_E_b)]
    # k=0
    # for w in range(no_of_w):
    #     for w_hat in range(no_of_w_hat):
    #         for E_b in range(no_of_E_b):
    #             states[k] = (w, w_hat, E_b)
    #             k += 1
    
    states = [(i,j,k) for i in range(no_of_w) for j in range(no_of_w_hat) for k in range(no_of_E_b)]
                
    return states

def driver():
    states = [0, 1, 2]
    transitions = [[0.1,0.1 ,0.8 ], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
    P = [[0, 0, 100], [0, 0, 100], [0, 0, 0]]
    
    mp = MP(states, transitions)
    bat = Battery(3, 1)
    
    states_arr = states_generator(len(mp.states), len(mp.states), bat.E_b_max + 1)
    actions_arr = [0,1]
    decider = Decider(states_arr, mp, P, bat, actions_arr, 0.9)
    
    decider.policy_iteration()
    
    print(decider.policy)
    # print(decider.value)
    
    

driver()