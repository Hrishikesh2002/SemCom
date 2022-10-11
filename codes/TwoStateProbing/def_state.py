class state:
    def __init__(self, last_sampled_value, last_sample_time ) -> None:
        self.last_sampled_value = last_sampled_value
        self.last_sample_time = last_sample_time

    def __repr__(self) -> str:
        return f"state(last_sampled_value={self.last_sampled_value}, last_sample_time={self.last_sample_time})"
