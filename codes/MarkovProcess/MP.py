
import random
import numpy as np

class MarkovProcess:
    def __init__(self, states, transitions) -> None:
        self.states = states
        self.transitions = np.array(transitions)
        
    def __repr__(self) -> str:
        return f"MarkovProcess(states={self.states}, transitions={self.transitions})"

    def sample_next_state(self, state: str) -> str:
        return random.choices(list(self.transitions[state].keys()), weights=list(self.transitions[state].values()))[0]
    
    def sample_path(self, state, path_length) -> list:
        path = [state]
        while state != 'TERMINAL' and len(path) < path_length:
            state = self.sample_next_state(state)
            path.append(state)
        return path
    
    


class MP:
    def __init__(self, states, transitions) -> None:
        self.states = states
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
    
    
def driver():
    states = ['A', 'B', 'C', 'D', 'E', 'TERMINAL']
    transitions = {'A':{ 'A':0.1, 'B':0.4, 'C':0.5}, 'B':{ 'A':0.4, 'B':0.1, 'C':0.5}, 'C':{ 'A':0.4, 'B':0.5, 'C':0.1}, 'D':{ 'D':0.1, 'E':0.9}, 'E':{ 'D':0.9, 'E':0.1}, 'TERMINAL':{ 'TERMINAL':1.0}}
    
    markovProcess = MarkovProcess(states, transitions)
    
    path = markovProcess.sample_path('A', 28)
    
    print(path)
    
    mp = MP([0,1], [[0.5,0.5],[0.5, 0.5]])
    
    path = mp.sample_path(0, 20)
    
    print(path)
    
if __name__ == "__main__":
    driver()
    
    

