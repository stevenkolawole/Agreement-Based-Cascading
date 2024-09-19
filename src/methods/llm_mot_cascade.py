from typing import List, Tuple, Union

from .coe import EnsembleCascade


class MOTLLMCascade(EnsembleCascade):
    def __init__(
        self,
        ServiceProvider,
        TaskData,
        cascade_tier_models: List[str],
        temperature: List[float] = [0.4, 0.6, 0.8],
        mixture_consistency_threshold: float = 2/3 # (2/3) or full consistency check = 1.0
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self._threshold = mixture_consistency_threshold
