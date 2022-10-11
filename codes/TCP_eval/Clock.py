#continuous clock for keeping track of packages arrival and departure times
import time


class Clock:
    def __init__(self) -> None:
        pass
    
    def start(self):
        self.start_time = time.time()
        # return self.get_time()
        
    def get_time(self):
        self.elapsed_time = time.time() - self.start_time
        return self.elapsed_time