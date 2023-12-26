from avstack.config import ALGORITHMS
from avstack.modules.perception.base import _PerceptionAlgorithm


@ALGORITHMS.register_module()
class GroundTruthLaneLineDetector(_PerceptionAlgorithm):
    MODE = "lane lines"

    def __call__(self, ground_truth, *args, **kwargs):
        """Wrap ground truths to detections"""
        return ground_truth.lane_lines


@ALGORITHMS.register_module()
class LaneNet(_PerceptionAlgorithm):
    pass
