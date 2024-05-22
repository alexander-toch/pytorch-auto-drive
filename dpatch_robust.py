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

from matplotlib.pylab import f
import numpy as np
from tqdm.auto import trange

from art.attacks.evasion import RobustDPatch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

import sys
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

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
            estimator=estimator, # type: ignore
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


    def _augment_images_with_patch(
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]], patch: np.ndarray, channels_first: bool
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Union[int, float]]]:
        """
        Augment images with patch.

        :param x: Sample images.
        :param y: Target labels.
        :param patch: The patch to be applied.
        :param channels_first: Set channels first or last.
        """

        transformations: Dict[str, Union[float, int]] = {}
        x_copy = x.copy() # (1, 288, 800, 3)
        patch_copy = patch.copy() # (patch_size[0], patch_size[1], 3)
        x_patch = x.copy()

        if channels_first:
            x_copy = np.transpose(x_copy, (0, 2, 3, 1))
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_copy = np.transpose(patch_copy, (1, 2, 0))

        # now we should be in the format (batch, height, width, channels)

        # Apply patch:        
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_copy.shape[1], y_1 + patch_copy.shape[0]

        # print(f"x_copy shape: {x_copy.shape}")
        # # print(f"patch_copy shape: {patch_copy.shape}")
        # print(f"x_patch shape: {x_patch.shape}")
        # print(f"patch_location: {self.patch_location}")
        # print(f"x_1: {x_1}, y_1: {y_1}, x_2: {x_2}, y_2: {y_2}")

        x_patch[:, y_1:y_2, x_1:x_2,  :] = patch_copy

        # # 1) crop images:
        # crop_x = random.randint(0, self.crop_range[0])
        # crop_y = random.randint(0, self.crop_range[1])
        # x_1, y_1 = crop_x, crop_y
        # x_2, y_2 = x_copy.shape[1] - crop_x + 1, x_copy.shape[2] - crop_y + 1
        # x_copy = x_copy[:, x_1:x_2, y_1:y_2, :]
        # x_patch = x_patch[:, x_1:x_2, y_1:y_2, :]

        # transformations.update({"crop_x": crop_x, "crop_y": crop_y})

        # 2) rotate images:
        rot90 = random.choices([0, 1, 2, 3], weights=self.rotation_weights)[0]

        x_copy = np.rot90(x_copy, rot90, (1, 2))
        x_patch = np.rot90(x_patch, rot90, (1, 2))

        transformations.update({"rot90": rot90})

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy = np.round(brightness * x_copy / self.learning_rate) * self.learning_rate
        x_patch = np.round(brightness * x_patch / self.learning_rate) * self.learning_rate

        transformations.update({"brightness": brightness})

        logger.debug("Transformations: %s", str(transformations))

        if self.targeted:
            predictions = y
        else:
            if channels_first:
                x_copy = np.transpose(x_copy, (0, 3, 1, 2))
            predictions = self.estimator.predict(x=x_copy, standardise_output=True)

        if channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch, predictions, transformations

    def generate(  # type: ignore
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]] = None, **kwargs
    ) -> tuple:
        """
        Generate RobustDPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if y is None and self.targeted:
            raise ValueError("The targeted version of RobustDPatch attack requires target labels provided to `y`.")
        if y is not None and not self.targeted:
            raise ValueError("The RobustDPatch attack does not use target labels.")
        if x.ndim != 4:  # pragma: no cover
            raise ValueError("The adversarial patch can only be applied to images.")

        # Check whether patch fits into the cropped images:
        if self.estimator.channels_first:
            image_height, image_width = x.shape[2:4]
        else:
            image_height, image_width = x.shape[1:3]

        if not self.estimator.native_label_is_pytorch_format and y is not None:
            from art.estimators.object_detection.utils import convert_tf_to_pt

            y = convert_tf_to_pt(y=y, height=x.shape[1], width=x.shape[2])

        # if (  # pragma: no cover
        #     self.patch_location[0] + self.patch_shape[0] > image_height - self.crop_range[0]
        #     or self.patch_location[1] + self.patch_shape[1] > image_width - self.crop_range[1]
        # ):
        #     raise ValueError("The patch (partially) lies outside the cropped image.")

        t = trange(self.max_iter, desc="RobustDPatch iteration", disable=not self.verbose)
        for i_step in t:
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients_old = np.zeros_like(self._patch)

            for e_step in range(self.sample_size):
                # if e_step == 0 or (e_step + 1) % 100 == 0:
                #     logger.info("EOT Step: %i", e_step + 1)

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size, x.shape[0])

                    if self.targeted:
                        y_batch = y['out'] # TODO: if batches should work, this needs to be fixed
                    else:
                        y_batch = y                  

                    # Sample and apply the random transformations:
                    patched_images, patch_target, transforms = self._augment_images_with_patch(
                        x[i_batch_start:i_batch_end], y_batch, self._patch, channels_first=self.estimator.channels_first
                    )

                    gradients, loss = self.estimator.loss_gradient(
                        x=patched_images,
                        y=patch_target,
                        standardise_output=True,
                    )
                    gradients = self._untransform_gradients(
                        gradients, transforms, channels_first=self.estimator.channels_first
                    )


                    # TODO: color optimization (optimise patch to look as much as possible like the original image OR grayscale)


                    patch_gradients = patch_gradients_old + np.sum(gradients, axis=0)
                    logger.debug(
                        "Gradient percentage diff: %f)",
                        np.mean(np.sign(patch_gradients) != np.sign(patch_gradients_old)),
                    )
                    t.set_postfix(loss=loss)
                    # t.set_postfix(diff=np.mean(np.sign(patch_gradients) != np.sign(patch_gradients_old)))

                    patch_gradients_old = patch_gradients

            # Write summary
            x_patched, y_patched, _ = self._augment_images_with_patch(
                x, y, self._patch, channels_first=self.estimator.channels_first
            )
            if self.summary_writer is not None:  # pragma: no cover

                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_step,
                    grad=np.expand_dims(patch_gradients, axis=0),
                    patch=self._patch,
                    estimator=self.estimator,
                    x=x_patched,
                    y=y_patched,
                    targeted=self.targeted,
                )

            self._patch = self._patch + np.sign(patch_gradients) * (1 - 2 * int(self.targeted)) * self.learning_rate


            # p = self._patch.transpose((1, 2, 0))
            # import cv2
            # p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)

            # # TODO: set to original pixel color
            # # save numpy array and try in jupyter notebook?
            
            # # ret, thresh = cv2.threshold(p, 0, 10, cv2.THRESH_BINARY)
            # # print(p.shape)
            # # print(p[thresh == 10])
            # # p[thresh == 10] = 128

            # p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)


            # p = p.transpose((2, 0, 1))

            # self._patch = p

            if self.estimator.clip_values is not None:
                self._patch = np.clip(
                    self._patch,
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return self._patch, x_patched, y_patched
    
    def _untransform_gradients(
        self,
        gradients: np.ndarray,
        transforms: Dict[str, Union[int, float]],
        channels_first: bool,
    ) -> np.ndarray:
        """
        Revert transformation on gradients.

        :param gradients: The gradients to be reverse transformed.
        :param transforms: The transformations in forward direction.
        :param channels_first: Set channels first or last.
        """

        if channels_first:
            gradients = np.transpose(gradients, (0, 2, 3, 1))

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo rotations:
        rot90 = int((4 - transforms["rot90"]) % 4)
        gradients = np.rot90(gradients, k=rot90, axes=(1, 2)) # shape: (1, 288, 800, 3) (1, H, W, C)

        # Account for cropping when considering the upper left point of the patch:
        x_1, y_1 = self.patch_location
        if channels_first:
            x_2 = x_1 + self.patch_shape[2]
            y_2 = y_1 + self.patch_shape[1]
        else:
            x_2 = x_1 + self.patch_shape[0]
            y_2 = y_1 + self.patch_shape[1]
        gradients = gradients[:, y_1:y_2, x_1:x_2, :]

        if channels_first:
            gradients = np.transpose(gradients, (0, 3, 1, 2))

        return gradients