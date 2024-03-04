# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements the adversarial patch attack `AdversarialPatch`. This attack generates an adversarial patch that
can be printed into the physical world with a common printer. The patch can be used to fool image and video estimators.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.evasion import AdversarialPatchPyTorch
from art.attacks.evasion.adversarial_patch.utils import insert_transformed_patch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.utils import check_and_transform_label_format, is_probability, to_categorical
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyAdversarialPatchPyTorch(AdversarialPatchPyTorch):
    

    def __init__(
        self,
        estimator: "CLASSIFIER_NEURALNETWORK_TYPE",
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Tuple[int, int, int] = (3, 224, 224),
        patch_location: Optional[Tuple[int, int]] = None,
        patch_type: str = "circle",
        optimizer: str = "Adam",
        targeted: bool = True,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ):
       super().__init__(estimator=estimator, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max, distortion_scale_max=distortion_scale_max, learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size, patch_shape=patch_shape, patch_location=patch_location, patch_type=patch_type, optimizer=optimizer, targeted=targeted, summary_writer=summary_writer, verbose=verbose)

    def _train_step(
        self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        import torch

        self.estimator.model.zero_grad()
        
        loss = self._loss(images, target, mask)

        loss.backward(retain_graph=True)

        if self._optimizer_string == "pgd":
            if self._patch.grad is not None:
                gradients = self._patch.grad.sign() * self.learning_rate
            else:
                raise ValueError("Gradient term in PyTorch model is `None`.")

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch + gradients, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
                )
        else:
            self._optimizer.step()

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
                )

        return loss

    def _predictions(
        self, images: "torch.Tensor", mask: Optional["torch.Tensor"], target: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        patched_input = self._random_overlay(images, self._patch, mask=mask)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        # import cv2
        # image = patched_input.detach().cpu().numpy()[0].transpose((1, 2, 0))
        # image = (image * 255).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("test.jpg", image)

        predictions, target = self.estimator._predict_framework(patched_input, target)  # pylint: disable=W0212

        return predictions['out'], target

    def _loss(self, images: "torch.Tensor", target: "torch.Tensor", mask: Optional["torch.Tensor"]) -> "torch.Tensor":
        import torch

        if isinstance(target, torch.Tensor):
            
            predictions, target = self._predictions(images, mask, target)

            prob_maps = torch.nn.functional.interpolate(predictions, size=(288, 800), mode='bilinear', align_corners=True) # TODO: set size dynamically 

            # print(f"prob_maps shape in adversarial_patch_pytorch.py: {prob_maps.shape}")
            # print(f"target shape in adversarial_patch_pytorch.py: {target.shape}")

            if self.use_logits:
                loss = torch.nn.functional.cross_entropy(
                    input=prob_maps, target=target, reduction="mean", weight=torch.tensor([0.4, 1, 1, 1, 1]).to(self.estimator.device)
                ) # TODO: get weight dynamically
                
            else:
                loss = torch.nn.functional.nll_loss(
                    input=prob_maps, target=torch.argmax(target, dim=1), reduction="mean"
                )

        else:
            assert False and "This should not happen"

        if (not self.targeted and self._optimizer_string != "pgd") or self.targeted and self._optimizer_string == "pgd":
            loss = -loss

        return loss


    def generate(  # type: ignore
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: An array with the original input images of shape NCHW or input videos of shape NFCHW.
        :param y: An array with the original true labels.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :return: An array with adversarial patch and an array of the patch mask.
        """
        import torch
        print(f"Device {self.estimator.device}")

        shuffle = kwargs.get("shuffle", True)
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)

        if self.patch_location is not None and mask is not None:
            raise ValueError("Masks can only be used if the `patch_location` is `None`.")

        if y is None:  # pragma: no cover
            logger.info("Setting labels to estimator predictions and running untargeted attack because `y=None`.")
            y = self.estimator.predict(x, batch_size=self.batch_size)
        else:
            y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

        if hasattr(self.estimator, "nb_classes"):
            # check if logits or probabilities

            if is_probability(y): # type: ignore
                self.use_logits = False
            else:
                self.use_logits = True

        if isinstance(y, np.ndarray):
            if self.estimator.nb_classes > 2:
               y_array = y / np.sum(y, axis=1, keepdims=True)

            x_tensor = torch.Tensor(x)
            y_tensor = torch.Tensor(y)


            if mask is None:
                dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
            else:
                mask_tensor = torch.Tensor(mask)
                dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, mask_tensor)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
        else:
            assert False and "This should not happen"
            
        t = trange(self.max_iter, desc="Adversarial Patch PyTorch", disable=not self.verbose)
        for i_iter in t:
            if mask is None:
                for images, target in data_loader:
                    images = images.to(self.estimator.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.estimator.device)
                    else:
                        assert False and "This should not happen"
                    loss = self._train_step(images=images, target=target, mask=None)
                    t.set_postfix(loss=loss.item())

            else:
               for images, target, mask_i in data_loader:
                    images = images.to(self.estimator.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.estimator.device)
                    else:
                        assert False and "This should not happen"
                    # mask_i = mask_i.to(self.estimator.device) # TODO: use GPU
                    _ = self._train_step(images=images, target=target, mask=mask_i)

        x_patched = (
            self._random_overlay(images=torch.from_numpy(x).to(self.estimator.device), patch=self._patch.detach())
            .detach()
            .cpu()
            .numpy()
        )

        return (
            self._patch.detach().cpu().numpy(),
            self._get_circular_patch_mask(nb_samples=1).cpu().numpy()[0],
            x_patched[0]
        )