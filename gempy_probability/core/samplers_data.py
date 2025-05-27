import dataclasses


@dataclasses.dataclass
class NUTS_Args:
    step_size: float = 0.0085
    adapt_step_size: bool = True
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    init_strategy: str = 'auto'
    num_samples: int = 200 
    warmup_steps: int = 50