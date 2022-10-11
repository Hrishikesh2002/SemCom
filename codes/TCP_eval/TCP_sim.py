from Clock import Clock

class state:
    def __init__(self):
        self.num_unack = 0
        self.time_since_ack = 0
        self.latency = 0


class Simulation:
    
    def __init__(self, state):
        self.state = state
        self.clock = Clock()
        self.clock.start()
        
    
        
    def updateState(self, action):
        if action:
            self.state.num_unack += 1
            self.state.time_since_ack = 0
        else:
            self.state.time_since_ack += 1
        
        
def driver():
    curr_state = state()
    sim = Simulation(curr_state)
    
    
    
        