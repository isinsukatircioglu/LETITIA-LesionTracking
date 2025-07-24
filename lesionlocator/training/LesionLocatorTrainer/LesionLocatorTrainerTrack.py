from typing import Union, Tuple, List
from torch import nn

from unigradicon import make_network, get_unigradicon
from lesionlocator.modules.tracknet import TrackNet
from lesionlocator.utilities.get_network_from_plans import get_network_from_plans


class LesionLocatorTrainerTrack():
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: Tuple[int, int, int],
                                   enable_deep_supervision: bool = True) -> nn.Module:
        unet = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels+1,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        
        # Registration
        reg_input_shape = [1, 1, 175, 175, 175]
        reg_net = make_network(reg_input_shape, include_last_step=True)
        # reg_net = get_unigradicon() # Also loads pre-trained weights, for training

        return TrackNet(reg_net, reg_input_shape[2:], unet, unet_patch_size=patch_size)