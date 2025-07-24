from typing import Union, Tuple, List
from torch import nn

from lesionlocator.utilities.get_network_from_plans import get_network_from_plans


class LesionLocatorTrainerSegment():
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        network = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels+1,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        return network