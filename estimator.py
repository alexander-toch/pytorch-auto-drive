from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch
from art.estimators.classification.pytorch import PyTorchClassifier
import logging
from art.utils import check_and_transform_label_format
if TYPE_CHECKING:
    # pylint: disable=C0412, C0302
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
    
logger = logging.getLogger(__name__)

class MyPyTorchClassifier(PyTorchClassifier):
    def __init__(self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: Optional[Union[float, str]] = "dynamic",
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
    ) -> None:
        super(MyPyTorchClassifier, self).__init__(model, loss, input_shape, nb_classes, optimizer, use_amp, opt_level, loss_scale, channels_first, clip_values, preprocessing_defences, postprocessing_defences, preprocessing, device_type)

    def loss_gradient(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        training_mode: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of the same shape as `x`.
        """
        import torch

        print(f"sample input shape: {x.shape}")
        print(f"y shape: {y.shape}")

        self._model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self.is_rnn:
            self._model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self.set_batchnorm(train=False)
                self.set_dropout(train=False)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                x_grad = x.clone().detach().requires_grad_(True)
            else:
                x_grad = torch.tensor(x).to(self._device)
                x_grad.requires_grad = True
            if isinstance(y, torch.Tensor):
                y_grad = y.clone().detach()
            else:
                y_grad = torch.tensor(y).to(self._device)
            inputs_t, y_preprocessed = self._apply_preprocessing(x_grad, y=y_grad, fit=False, no_grad=False)
        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            x_grad.requires_grad = True
            inputs_t = x_grad
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)
        else:
            labels_t = y_preprocessed

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        #  def forward(self, inputs: Tensor, targets: Tensor, net, interp_size):
        loss = self._loss(model_outputs[-1], labels_t, self._model)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        if self._use_amp:  # pragma: no cover
            from apex import amp  # pylint: disable=E0611

            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            loss.backward()

        if x_grad.grad is not None:
            if isinstance(x, torch.Tensor):
                grads = x_grad.grad
            else:
                grads = x_grad.grad.cpu().numpy().copy()
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads
    
    def _predict_framework(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :return: Tensor of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=False)

        # Put the model in the eval mode
        self._model.eval()

        model_outputs = self._model(x_preprocessed)
        output = model_outputs[-1]

        return output, y_preprocessed
    
    def compute_loss(  # type: ignore # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        reduction: str = "none",
        **kwargs,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Array of losses of the same shape as `x`.
        """
        import torch

        # x shape should be (batch_size, channels, height, width)
        # y shape should be (batch_size, nb_classes, height, width)

        self._model.eval()

        # y = check_and_transform_label_format(y, self.nb_classes)  # type: ignore

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        y = self.reduce_labels(y_preprocessed)


        if isinstance(x, torch.Tensor):
            inputs_t = x_preprocessed
        else:
            # Convert the inputs to Tensors
            inputs_t = torch.from_numpy(x_preprocessed).to(self._device)

        if isinstance(y, torch.Tensor):
            y_grad = y.clone().detach()
        else:
            y_grad = torch.tensor(y).to(self._device)

        # Compute the loss and return
        model_outputs = self._model(inputs_t)
        prev_reduction = self._loss.reduction

        # Return individual loss values
        self._loss.reduction = reduction
        loss = self._loss(model_outputs[-1], y_grad, self._model)
        self._loss.reduction = prev_reduction

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()