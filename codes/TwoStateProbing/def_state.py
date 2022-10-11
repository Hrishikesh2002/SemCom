class state:
    def __init__(self, last_sampled_value, last_sample_time ) -> None:
        self.last_sampled_value = last_sampled_value
        self.last_sample_time = last_sample_time

    def __repr__(self) -> str:
        return f"state(last_sampled_value={self.last_sampled_value}, last_sample_time={self.last_sample_time})"


class alg_state:
    def __init__(self, last_sampled_state, last_sample_time , curr_state_cost) -> None:
        self.last_sampled_state = last_sampled_state
        self.last_sample_time = last_sample_time
        self.curr_state_cost = curr_state_cost
        
    def __repr__(self) -> str:
        return f"state(last_sampled_state={self.last_sampled_state}, last_sample_time={self.last_sample_time}, prev_state_cost={self.prev_state_cost})"