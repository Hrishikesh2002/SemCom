#This would be an efficent implementation than model 1, but needs to be fixed.

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
		if(self.E_b == 0):
			raise Exception("Battery is empty")
		self.E_b -= 1
		
	def charge(self):
		if self.rv.rvs() == 1 and self.E_b < self.E_b_max:
			self.E_b += 1
			

class Decider:
	def __init__(self, states:list, process:MP, P:list, bat:Battery, actions:list = [0,1], discount_factor:int=0.8):
		self.states = states 
		self.bat = bat
		self.process = process
		self.num_states = (len(self.process.states), len(self.process.states), self.bat.E_b_max + 1)
		self.discount_factor = discount_factor
		self.actions = actions
		self.last_transmitted_value = 0       # Last predicted value
		self.P = P

		self.define_reward()
		self.define_transitions()
		
		
		
	#correct
	def define_reward(self):
		reward = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2], len(self.actions)))
  
		#Iterate over all states and assign the value of -P(w,w_hat) to the reward
		for i1 in range(self.num_states[0]):
			for i2 in range(self.num_states[1]):
				for i3 in range(self.num_states[2]):
					for j in range(len(self.actions)):
						reward[i1,i2,i3,j] = -self.P[self.states[i1,i2,i3,0]][self.states[i1,i2,i3,1]]
      
		print("Reward = ", reward)
						
		self.reward = reward
		
	def define_transitions(self):
		self.transitions = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2], len(self.actions), self.num_states[0], self.num_states[1], self.num_states[2]))
		for i1 in range(self.num_states[0]):
			for i2 in range(self.num_states[1]):
				for i3 in range(self.num_states[2]):
					for j in range(len(self.actions)):
						
							for j1 in range(self.num_states[0]):
								w_transition_prob = self.process.transitions[i1,j1]
								w_hat_arr = np.zeros(self.num_states[1])
								w_hat_arr[i2] = 1
								bat_arr = np.zeros(self.num_states[2])
								if self.actions[j]:
									if i3 == self.bat.E_b_min:  #The battery is empty. Ideally this should not happen
										bat_arr[i3] = 1
									else:
										bat_arr[i3] = self.bat.p
										bat_arr[i3 - 1] = 1-self.bat.p
										

								else:
									if i3 == self.bat.E_b_max:
										bat_arr[i3] = 1
									else:
										bat_arr[i3] = 1 - self.bat.p
										bat_arr[i3 + 1] = self.bat.p

								self.transitions[i1,i2,i3,j,j1] = np.outer(w_hat_arr, bat_arr) * w_transition_prob

     
	def value_iteration(self, epsilon=0.0001):
		# Initialize value function
		V = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2]))
		# Initialize policy
		policy = np.zeros((self.num_states[0], self.num_states[1], self.num_states[2]))
		# Initialize iteration counter
		num_iterations = 0
		# Repeat until convergence
		while True:
			# Initialize delta
			delta = 0
			# Update each state
			for i1 in range(self.num_states[0]):
				for i2 in range(self.num_states[1]):
					for i3 in range(self.num_states[2]):
         
						# Do a one-step lookahead to find the best action
						action_values = np.zeros(len(self.actions))
						for j in range(len(self.actions)):
							for j1 in range(self.num_states[0]):
								for j2 in range(self.num_states[1]):
									for j3 in range(self.num_states[2]):
										action_values[j] += self.transitions[i1,i2,i3,j,j1,j2,j3] * (self.reward[i1,i2,i3,j] + self.discount_factor * V[j1,j2,j3])
						# Select the best action
						best_action_value = np.max(action_values)
						# Calculate delta across all states seen so far
						delta = max(delta, np.abs(best_action_value - V[i1][i2][i3]))
						# Update the value function. Ref: Sutton book eq. 4.10.
						V[i1][i2][i3] = best_action_value
						# Greedily update the policy
						policy[i1][i2][i3] = self.actions[np.argmax(action_values)]
			# Check if we can stop 
			num_iterations += 1
			if delta < epsilon:
				break

		return policy, V
	
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
		num_actions = len(self.actions)
		for i in range(self.num_states[0]):
			for j in range(self.num_states[1]):
				for k in range(self.num_states[2]):
					policy[i][j][k] = np.ones(num_actions)/num_actions
					
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
    
				# Iterate over all states
				for i1 in range(self.num_states[0]):
					for i2 in range(self.num_states[1]):
						for i3 in range(self.num_states[2]):
          
							# pick one action
							for j in range(len(self.actions)):
								pi = policy[i1,i2,i3,j]
        
								#Take expectation over all states leading from that action
								environmental_expectation = np.dot(self.transitions[i1,i2,i3,j], value)
        
								#Add the expected reward for that action to the value of the state
								new_value[i] += pi * (self.reward[i1,i2,i3,j] + self.discount_factor * environmental_expectation)
								
								
								
				if np.allclose(new_value, value, atol=1e-10):
					break
				
				value = new_value
			
			#policy improvement

	
			#Iterate over all states		
			for i1 in range(self.num_states[0]):
				for i2 in range(self.num_states[1]):
					for i3 in range(self.num_states[2]):
         
						#Find the best action(s) (i.e. find the action(s) with the highest value)
						max_value = -np.inf
						action_reward_list = []
						for j in range(len(self.actions)):
							state_action_value = self.reward[i1,i2,i3,j] 
							action_reward_list.append(state_action_value)
							if(state_action_value > max_value):
								max_value = state_action_value


						#Initialize the new policy
						new_policy[i1,i2,i3] = np.zeros(len(self.actions))
      
						#Find the number of actions with the highest value
						num_max_actions = action_reward_list.count(max_value)
						
						#Assign the probability of taking the best action(s) to the new policy (greedy step)
						for k in range(len(self.actions)):
							if(action_reward_list[k] == max_value):
								new_policy[i1,i2,i3,k] = 1/num_max_actions

								
			#check if policy is stable
			if np.allclose(policy, new_policy, atol=1e-10):
				policy_stable = True
			
			policy = new_policy
			
		return policy, value
	




def states_generator(w_max, w_hat_max, E_b_max):
	states = []
	for i in range(w_max+1):
		w_list = []
		for j in range(w_hat_max+1):
			w_hat_list = []
			for k in range(E_b_max+1):
				w_hat_list.append((i, j, k))
			w_list.append(w_hat_list)
		states.append(w_list)
  
	states = np.array(states)
 
	print("States = ", states)
	return states


def driver():
    states = [0, 1, 2]
    transitions = [[0.1, 0.7, 0.2], [0.2, 0.6, 0.2], [0.3, 0.5, 0.2]]
    P = [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
    
    mp = MP(states, transitions)
    
    bat = Battery(5, 3, 0, 0.5)
    
    decider = Decider(states=states_generator(2, 2, 5), bat=bat, process=mp, P=P, actions=[0, 1], discount_factor=0.9)
    
    policy, value =decider.value_iteration()
    
    print("The final policy after value iteration: \n ", policy)
    

if __name__ == "__main__":
	driver()


