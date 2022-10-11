import numpy as np
import random

class MP:
    def __init__(self, states, transitions) -> None:
        self.states = states
        self.num_states = len(states)
        self.transitions = np.array(transitions)
        
    
    def __repr__(self) -> str:
        return f"MarkovProcess(states={self.states}, transitions={self.transitions})"
    
    def sample_next_state(self, state:int) -> int:
        return random.choices(self.states, weights=self.transitions[state])[0]
    
    def sample_path(self, state, path_length) -> list:
        path = [state]
        while len(path) < path_length:
            state = self.sample_next_state(state)
            path.append(state)
        
        return path


