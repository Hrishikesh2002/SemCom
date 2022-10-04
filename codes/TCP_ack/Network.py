import time
import scipy.stats as dist
import matplotlib.pyplot as plt
import numpy as np

class Clock:
    def __init__(self) -> None:
        pass
    
    def start(self):
        self.start_time = time.time()
        # return self.get_time()
        
    def get_time(self):
        self.elapsed_time = time.time() - self.start_time
        return self.elapsed_time
    

class Packet:
    id = 0
    def __init__(self, generation_time, transmission_time, arrival_time, acknowledged_time, size, source=0, destination=1) -> None:
        self.generation_time = generation_time
        self.transmission_time = transmission_time
        self.arrival_time = arrival_time
        self.acknowledged_time = acknowledged_time
        self.size = size
        self.source = source
        self.destination = destination
        self.id = Packet.id
        Packet.id += 1
        
    def __repr__(self) -> str:
        return f"Packet(id:{self.id}, gen:{self.generation_time}, tran:{self.transmission_time}, arr:{self.arrival_time}, ack:{self.acknowledged_time})"
        


class Transmittor:
    def __init__(self, process, clock) -> None:
        self.process = self.select_process(process)
        self.queue = []
        self.transmitted_packets = []
        self.clock = clock
        
        
        
    def select_process(self, process):
        if process == 0:
            return dist.bernoulli(0.5)
        elif process == 1:
            return dist.poisson(0.5)
        else:
            raise Exception("Process not defined")
        
    
    def generate_packet(self):
        if self.process.rvs() == 1:
            self.queue.append(Packet(self.clock.get_time(), 0, 0, 0, np.random.uniform(0, 60)))
            
    def transmit_packet(self):
        if len(self.queue) > 0:
            if(np.random.uniform(0, 1) < 0.5):
                self.queue[0].transmission_time = self.clock.get_time()
                return self.queue.pop(0)
            else:
                return None
        
        else:
            return None
        
        
class Reciever:
    def __init__(self, clock) -> None:
        self.queue = []
        self.received_packets = []
        self.clock = clock
        self.last_acknowledged_time = 0
        self.last_acknowledged_id = -1
        
        
        
    def receive_packet(self, packet):
        if(packet):
            packet.arrival_time = self.clock.get_time()
            self.queue.append(packet)
            

        
    def acknowledge_packet(self, packet):
        self.last_received_time = packet.acknowledged_time = self.clock.get_time()
        self.last_acknowledged_id = packet.id
        
        
        
        
class MDP_acknowledger:
    
    def __init__(self, receiver) -> None:
        self.receiver = receiver
        self.states = self.generate_states
        self.transitions = self.generate_transitions
        self.reward = self.generate_reward
        self.actions = [0, 1]
    
    def generate_states(self):
        num_packets = len(self.receiver.queue)
        
        
        
        
class Algorithm_acknowledger:
    def __init__(self, receiver, z=0.00000000003) -> None:
        self.receiver = receiver
        self.z = z
    
    
    def calculate_packets_in_interval(self, t, t_p):
        num_packets = 0
        for packet in self.receiver.queue:
            if packet.arrival_time > t and packet.arrival_time < t_p:
                num_packets += 1
        return num_packets
    
    
    def decide(self):
        if len(self.receiver.queue) > 0 and self.receiver.last_acknowledged_id != self.receiver.queue[-1].id:
            
            start = self.receiver.last_acknowledged_time
            end = self.receiver.clock.get_time()

            offset = (start - end)/10

            for i in range(10):
                if self.calculate_packets_in_interval(start + offset*i, start + offset*(i+1)) * (offset*(9-i)) >= self.z:
                    for i in range(self.receiver.last_acknowledged_id, self.receiver.queue[-1].id + 1):
                        self.receiver.acknowledge_packet(self.receiver.queue[i])
                        
                    self.receiver.last_acknowledged_time = self.receiver.clock.get_time()
                    self.receiver.last_acknowledged_id = self.receiver.queue[-1].id
                    break
                

            
            
        
    

      
    

class TCP_simulation:
    
    def __init__(self, process, clock, ack=0) -> None:
        self.transmittor = Transmittor(process, clock)
        self.reciever = Reciever(clock)
        self.clock = clock
        self.current_time = 0
        self.current_state = 0
        self.next_state = 0
        self.ack = ack
        
    def run(self):
        
        if(self.ack == 0):
            self.acknowledger = Algorithm_acknowledger(self.reciever)
        else:
            self.acknowledger = MDP_acknowledger(self.reciever)
            
            
        self.transmittor.clock.start()
        self.reciever.clock.start()
        self.clock.start()


        while self.current_time < 0.01:

            self.transmittor.generate_packet()
            self.reciever.receive_packet(self.transmittor.transmit_packet())
            self.acknowledger.decide()
            self.current_time = self.clock.get_time() 
            
        for packets in self.reciever.queue:
            print(packets)
        
            
                   
                
       
def driver():
    clock = Clock()
    sim = TCP_simulation(0, clock, 0)
    sim.run()
    
driver()