import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F

from ._utils import WeightedLoss, _Loss
from .builder import LOSSES


# Typical lane detection loss by binary segmentation (e.g. SCNN)
@LOSSES.register()
class LaneLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: float = 0.1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean'):
        super(LaneLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.existence_weight = existence_weight

    def forward(self, inputs: Tensor, targets: Tensor, lane_existence: Tensor, net, interp_size):
        outputs = net(inputs)
        prob_maps = torch.nn.functional.interpolate(outputs['out'], size=interp_size, mode='bilinear',
                                                    align_corners=True)
        targets[targets > lane_existence.shape[-1]] = 255  # Ignore extra lanes
        segmentation_loss = F.cross_entropy(prob_maps, targets, weight=self.weight,
                                            ignore_index=self.ignore_index, reduction=self.reduction)
        existence_loss = F.binary_cross_entropy_with_logits(outputs['lane'], lane_existence,
                                                            weight=None, pos_weight=None, reduction=self.reduction)
        total_loss = segmentation_loss + self.existence_weight * existence_loss

        return total_loss, {'training loss': total_loss, 'loss seg': segmentation_loss,
                            'loss exist': existence_loss}

@LOSSES.register()
class LaneLossSeg(_Loss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: float = 0.1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean'):
        super(LaneLossSeg, self).__init__(size_average, reduce, reduction)
        self.weight = torch.Tensor(weight) if weight is not None else None
        self.ignore_index = ignore_index
       
    def forward(self, inputs: Tensor, targets: Tensor, input_sizes):
        prob_maps = torch.nn.functional.interpolate(inputs['out'], size=input_sizes, mode='bilinear', align_corners=True)
       
        # self.save_prob_map_images(prob_maps.softmax(dim=1))

        targets[targets > 5] = 255  # Ignore extra lanes TODO: set dynamically

        # print(f"prob_maps shape: {prob_maps.shape}")
        # print(f"targets shape: {targets.shape}")

        if targets.shape is not prob_maps.shape:
            targets = torch.nn.functional.interpolate(targets, size=input_sizes, mode='bilinear', align_corners=True) 

        loss_type = 'cross_entropy_argmax'  # TODO: check what which loss makes the most sense, or if armgax is the right thing to use
        if loss_type == 'cross_entropy_argmin':
            segmentation_loss = F.cross_entropy(prob_maps, torch.argmin(targets, dim=1), weight=self.weight.cuda(), reduction=self.reduction, ignore_index=self.ignore_index)
        elif loss_type == 'cross_entropy_argmax':
            segmentation_loss = F.cross_entropy(prob_maps, torch.argmax(targets, dim=1), weight=self.weight.cuda(), reduction=self.reduction, ignore_index=self.ignore_index)
        elif loss_type == 'nll':
            segmentation_loss = -F.nll_loss(input=prob_maps, target=torch.argmax(targets, dim=1), reduction="mean")
        else:
            segmentation_loss = F.cross_entropy(prob_maps, targets, weight=self.weight.cuda(), reduction=self.reduction)

        return segmentation_loss

    def save_prob_map_images(prob_maps):
         # save prob map as image        
        import numpy as np
        for i, lane in enumerate(prob_maps[0]):
            pred =lane.detach().cpu().numpy()
            rescaled = (255.0/pred.max() * (pred - pred.min())).astype(np.uint8)
            from PIL import Image
            im = Image.fromarray(rescaled)
            im.save(f'prob_map_image_{i}.png')
    
# Loss function for SAD
@LOSSES.register()
class SADLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: float = 0.1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean'):
        super(SADLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.existence_weight = existence_weight

    def forward(self, inputs: Tensor, targets: Tensor):
        pass
