import random
import numpy as np

class MarkovDecisionProcess:
  def __init__(self, states, transitions, actions, discount_factor, reward ):

    self.states = states
    self.num_states = len(self.states)
    # transitions are s*t*s
    self.transitions = transitions
    self.discount_factor = discount_factor
    self.reward = reward
    self.actions = actions
    
  def sample_next_action(self, state, policy):
    return random.choice(self.actions ,weights=policy[state])
  
  def sample_next_state(self, state, action):
    return random.choice(self.states ,weights=self.transitions[state][action])
  
  def sample_path(self, state, path_length, policy):
    path = [state]
    i=0
    sum_reward = 0
    action_seq = []
    while len(path) < path_length:
      next_action = self.sample_next_action(path[-1], policy)
      next_state = self.sample_next_state(path[-1], next_action)
      action_seq.append(next_action)
      sum_reward += self.reward[path[-1]][action_seq[-1]]
      path.append(next_state)
    
    return path, sum_reward, action_seq
  
  
  def sample_paths(self, num_paths, path_length, policy):
    paths = []
    for i in range(num_paths):
      paths.append([self.sample_path(random.choice(self.states), path_length, policy)]) 
      
    return paths

  
  def policy_iteration(self):
    #initialize policy
    policy = np.zeros((self.num_states, len(self.actions)))
    for i in range(self.num_states):
      policy[i] = np.ones(len(self.actions))/len(self.actions)
    
    #initialize value function
    value = np.zeros(self.num_states)
    
    #initialize policy evaluation
    new_value = np.zeros(self.num_states)
    
    #initialize policy improvement
    new_policy = np.zeros((self.num_states, len(self.actions)))
    
    #initialize policy_stable
    policy_stable = False
    
    while not policy_stable:
      #policy evaluation
      while True:
        new_value = np.zeros(self.num_states)
        for i in range(self.num_states):
          for j in range(len(self.actions)):
            pi = policy[i][j]
            environmental_expectation = np.dot(self.transitions[i][j], value)
            new_value[i] += pi * (self.reward[i][j] + self.discount_factor * environmental_expectation)
            
        
        if np.allclose(value, new_value, atol=1e-10):
          break
        
        value = new_value
        
      #policy improvement
      
      for i in range(self.num_states):
        max_value = -np.inf
        action_reward_list = []
        for j in range(len(self.actions)):
          state_action_value = self.reward[i][j] + self.discount_factor * np.dot(self.transitions[i][j], value)
          action_reward_list.append(state_action_value)
          if(state_action_value > max_value):
            max_value = state_action_value
            
        new_policy[i] = np.zeros(len(self.actions))
        num_max_actions = np.where(action_reward_list == max_value)[0].shape[0]
            
        for k in range(len(self.actions)):
          if(action_reward_list[k] == max_value):
            new_policy[i][k] = 1/num_max_actions
            
      #check if policy is stable
      if np.allclose(policy, new_policy, atol=1e-10):
        policy_stable = True
        
      policy = new_policy
    
    return policy, value
          
        

  def __repr__(self) -> str:
    
    return f"MarkovDecisionProcess(states={self.states}, transitions={self.transitions}, actions={self.actions}, discount_factor={self.discount_factor}, reward={self.reward})"
  
  
def driver():
  states = [0,1,2,3,4]
  transitions = [[[0.2, 0.2, 0.2, 0.2, 0.2],[0.5, 0.5, 0, 0, 0]], [[0.2, 0.2, 0.2, 0.2, 0.2],[0.5, 0.5, 0, 0, 0]], [[0.2, 0.2, 0.2, 0.2, 0.2],[0.5, 0.5, 0, 0, 0]], [[0.2, 0.2, 0.2, 0.2, 0.2],[0.5, 0.5, 0, 0, 0]], [[0.2, 0.2, 0.2, 0.2, 0.2],[0.5, 0.5, 0, 0, 0]]]
  reward = [[0,0], [0,0], [0,0], [0,0], [0,1]]
  actions = [0,1]
  discount_factor = 0.9
  
  markovDecisionProcess = MarkovDecisionProcess(states, transitions, actions, discount_factor, reward)
  
  policy, value = markovDecisionProcess.policy_iteration()
  
  print("Policy: ", policy)
  print("Value: ", value)
  

if __name__=='__main__':
  driver()
  