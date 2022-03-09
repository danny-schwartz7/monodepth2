import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List
from collections import deque

from unsupervised.Blocks import ConvElu, UpConvElu, Reshaper, ResConv, UpsampleAndSmooth

MAX_DISP_FRAC = 0.3  # from the paper


class StereoEncDecNet(nn.Module):
    # take the place of the bowtie-looking thing in the diagrams of https://arxiv.org/pdf/1609.03677.pdf

    def __init__(self, mode: str = "upsample"):
        super(StereoEncDecNet, self).__init__()

        self._init_model(mode)

        # start with 3 x 375 x 1242
        # first, get down to a reasonable resolution

    def assert_mode(self, mode: str):
        assert mode in ["upsample", "upconv"]

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param x: an input tensor of shape (N, C, H, W) where N is batch size,
            C is # channels (e.g., RGB = 3), H and W are height and width
            IMPORTANT: RGB input is expected to be floating point between 0 and 1
        :return: a list of tuples of two disparity maps, left-to-right disp
            and right-to-left disp (dimensions determined by KITTI).
            These maps will have elements in [-1, 1] and can be transformed
            into a more conventional view of disparity by subtracting
            torch.linspace(-1, 1, width) from each column and then
            multiplying by width to get a number of pixels.

            Each tuple in the list corresponds to a different output scale. The list is ordered with
            ascending resolution (more granular predictions are toward the end of the list).
        """
        print("Passing tensor of shape ", x.shape)
        # use a deque to implement skip connections
        skip_tensor_deque = deque()

        for module in self.skip_down_blocks:
            x = module(x)
            skip_tensor_deque.append(x)

        for module in self.internal_blocks:
            x = module(x)

        disp_pairs_at_scales = []

        for i, module in enumerate(self.skip_up_blocks):
            skip = skip_tensor_deque.pop()
            if i > 1 and i < (len(self.skip_up_blocks) - 1):
                # assume 'module' is of type 'UpConvElu' or 'UpsampleAndSmooth'
                x = module.up(x + skip)
                disp_part = x[:, :2, :, :]
                disp_part = MAX_DISP_FRAC * F.sigmoid(disp_part)
                left_to_right_disp = disp_part[:, 0, :, :]
                right_to_left_disp = disp_part[:, 1, :, :]
                disp_pairs_at_scales.append((left_to_right_disp, right_to_left_disp))
                x = module.elu(x)
            else:
                x = module(x + skip)

        for module in self.smoothing_blocks:
            x = module(x)

        disp_maps = MAX_DISP_FRAC * F.sigmoid(x)

        left_to_right_disp = disp_maps[:, 0, :, :]
        right_to_left_disp = disp_maps[:, 1, :, :]

        disp_pairs_at_scales.append((left_to_right_disp, right_to_left_disp))
        # The largest scale is the last element of this list
        return disp_pairs_at_scales

    def _init_model(self, mode: str = "upsample"):
        self.skip_down_blocks = nn.ModuleList([
            ConvElu(6, 4, 5, 2),
            ConvElu(4, 4, 5, 2),
            ConvElu(4, 8, 5, 2),

            # after this point, we wouldn't extract depth maps
            ConvElu(8, 16, (3, 7), (2, 3)),
            ConvElu(16, 32, (3, 7), (2, 2)),
            ConvElu(32, 4, (3, 7), (1, 1))
        ])
        self.internal_blocks = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(512, 128),
            # TODO: weird issue when using batchnorm during training: fails when batch size is 1, can happen due to imperfect divisiblity of train set size
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            Reshaper((1, 8, 16)),
            UpConvElu(1, 4, 1, 1, 0, 0)
        ])

        if mode == "upconv":
            self.skip_up_blocks = nn.ModuleList([
                UpConvElu(4, 32, (3, 7), 1, 0, 0),
                UpConvElu(32, 16, (3, 7), 2, 0, 0),
                UpConvElu(16, 8, (3, 7), (2, 3), 0, 1),
                UpConvElu(8, 4, 5, 2, 0, (0, 1)),
                UpConvElu(4, 4, 5, 2, 0, (1, 0)),

                # non-smoothed output:
                # UpConvElu(4, 2, 5, 2, 0, (0, 1), activation=False)

                # smoothed output:
                UpConvElu(4, 4, 5, 2, 0, (0, 1))
            ])
        elif mode == "upsample":
            self.skip_up_blocks = nn.ModuleList([
                UpsampleAndSmooth(in_channels=4, out_channels=32, out_height=10, out_width=22, smooth_kernel_size=3),
                UpsampleAndSmooth(in_channels=32, out_channels=16, out_height=21, out_width=49, smooth_kernel_size=3),
                UpsampleAndSmooth(in_channels=16, out_channels=8, out_height=44, out_width=152),
                UpsampleAndSmooth(in_channels=8, out_channels=4, out_height=91, out_width=308),
                UpsampleAndSmooth(in_channels=4, out_channels=4, out_height=186, out_width=619),
                UpsampleAndSmooth(in_channels=4, out_channels=4, out_height=375, out_width=1242)
            ])
        else:
            raise ValueError(f"Invalid mode {mode}")
        self.smoothing_blocks = nn.ModuleList([
            ResConv(4, 7, 2),
            nn.Conv2d(4, 2, 7, padding="same")
        ])

