
from collections import defaultdict

from typing import Callable, Any, Dict

from tensorboardX import SummaryWriter


class RunningLossTracker:
    def __init__(self, tbx_writer: SummaryWriter, ingestor: Callable[[Any], Dict[str, float]]):
        self.tbx_writer: SummaryWriter = tbx_writer
        self.stats_dict: Dict[str, float] = defaultdict(lambda: 0)
        self.ingestor: Callable[[Any], Dict[str, float]] = ingestor
        self.denominator: int = 0

    def ingest(self, x: Any, denominator_update: int, multiplier: float = 1.0):
        self.denominator += denominator_update

        stats_update_dict = self.ingestor(x)

        for key in stats_update_dict.keys():
            self.stats_dict[key] += multiplier * stats_update_dict[key]

    def log(self, log_index: int):
        if self.denominator == 0.0:
            raise ValueError("Must ingest something before logging!")

        for key in self.stats_dict.keys():
            self.tbx_writer.add_scalar(key, self.stats_dict[key]/self.denominator, log_index)
            self.stats_dict[key] = 0.0

        self.denominator = 0
