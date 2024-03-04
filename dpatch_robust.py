# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements a variation of the adversarial patch attack `DPatch` for object detectors.
It follows Lee & Kolter (2019) in using sign gradients with expectations over transformations.
The particular transformations supported in this implementation are cropping, rotations by multiples of 90 degrees,
and changes in the brightness of the image.

| Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
| Paper link (physical-world patch from Lee & Kolter): https://arxiv.org/abs/1906.11897
"""
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.evasion import RobustDPatch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class MyRobustDPatch(RobustDPatch):
    def __init__(
        self,
        estimator: "CLASSIFIER_NEURALNETWORK_TYPE",
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        patch_location: Tuple[int, int] = (0, 0),
        crop_range: Tuple[int, int] = (0, 0),
        brightness_range: Tuple[float, float] = (1.0, 1.0),
        rotation_weights: Union[Tuple[float, float, float, float], Tuple[int, int, int, int]] = (1, 0, 0, 0),
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        targeted: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            estimator=estimator,
            patch_shape=patch_shape,
            patch_location=patch_location,
            crop_range=crop_range,
            brightness_range=brightness_range,
            rotation_weights=rotation_weights,
            sample_size=sample_size,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            targeted=targeted,
            summary_writer=summary_writer,
            verbose=verbose,
            **kwargs
        )