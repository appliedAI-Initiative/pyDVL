"""
This module provides an implementation of DeepSet, from Zaheer et al. (2017)...

DeepSet uses a simple permutation-invariant architecture to learn embeddings for
sets of points, see...

## References

...

"""

from __future__ import annotations

from typing import Any, Collection, cast

from typing_extensions import Self

try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("pytorch required to use DeepSet.") from e

import torch.nn as nn
import torch.nn.init as init
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import trange

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import Sample
from pydvl.valuation.utility.learning import UtilityModel

__all__ = ["DeepSet", "DeepSetUtilityModel"]


class DeepSet(nn.Module):
    r"""Simple implementation of DeepSets to learn utility functions.

    Given a set $S= \{x_1, x_2, ..., x_n\},$ deepset learns a representation of the set
    which is invariant to the order of elements in the set. The model consists of two
    networks:

    $$ \Phi(S) = \sum_{x_i \in S} \phi(x_i), $$

    where $\phi(x_i)$ is a learned embedding for data point $x_i,$ and a second network
    $\rho$ that predicts the output $y$ from the aggregated representation:

    $$ y = \rho(\Phi(S)). $$

    Args:
        input_dim: Dimensions of each instance in the set, or dimension of the embedding
            if using one.
        phi_hidden_dim: Number of hidden units in the phi network.
        phi_output_dim: Output dimension of the phi network.
        rho_hidden_dim: Number of hidden units in the rho network.
        use_embedding: If `True`, use an embedding layer to learn representations for
            x_i.
        num_embeddings: Number of unique x_i values (only needed if `use_embedding` is
            `True`).
    """

    def __init__(
        self,
        input_dim: int,
        phi_hidden_dim: int,
        phi_output_dim: int,
        rho_hidden_dim: int,
        use_embedding: bool = False,
        num_embeddings: int | None = None,
    ):
        super(DeepSet, self).__init__()

        self.use_embedding = use_embedding
        if use_embedding:
            if num_embeddings is None or input_dim is None:
                raise ValueError(
                    "num_embeddings and input_dim must be provided when using embedding"
                )
            self.embedding = nn.Embedding(num_embeddings, input_dim)

        # The phi network processes each element in the set individually.
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden_dim),
            nn.ReLU(),
            nn.Linear(phi_hidden_dim, phi_output_dim),
            nn.ReLU(),
        )
        # The rho network processes the aggregated (summed) representation.
        self.rho = nn.Sequential(
            nn.Linear(phi_output_dim, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif self.use_embedding and isinstance(module, nn.Embedding):
            init.xavier_uniform_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: If using embedding, x should be of shape (batch_size, set_size) with
                integer ids. Otherwise, x is of shape (batch_size, set_size, input_dim)
                with feature vectors.
        Returns:
            Output tensor of shape (batch_size, 1), the predicted y for each set.
        """
        if self.use_embedding:
            x = self.embedding(x)  # shape (batch_size, set_size, embed_dim)

        phi_x = self.phi(x)  # shape: (batch_size, set_size, phi_output_dim)
        aggregated = torch.sum(phi_x, dim=1)  # shape: (batch_size, phi_output_dim)
        out = self.rho(aggregated)  # shape: (batch_size, 1)
        return cast(Tensor, out)


class SetDatasetRaw(TorchDataset):
    def __init__(
        self,
        samples: dict[Sample, float],
        training_data: Dataset,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            samples: Mapping from samples to target y.
            training_data: the [Dataset][pydvl.valuation.dataset.Dataset] from which the
                samples are drawn.

        """
        self.dtype = dtype
        self.device = device
        self.samples = list(samples.items())
        self.data = training_data
        self.max_set_size = max(len(s.subset) for s, _ in self.samples)  # For padding

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Builds the tensor for the set with index `idx`"""
        sample, y = self.samples[idx]
        set_tensor = torch.zeros(
            self.max_set_size,
            self.data.n_features,
            dtype=self.dtype,
            device=self.device,
        )
        for i, idx in enumerate(sample.subset):
            set_tensor[i] = set_tensor.new_tensor(self.data.data().x[idx])
        return set_tensor, set_tensor.new_tensor([y])


class DeepSetUtilityModel(UtilityModel):
    """
    A utility model that uses a simple DeepSet architecture to learn utility functions.

    Args:
        data: The pydvl dataset from which the samples are drawn.
        phi_hidden_dim: Number of hidden units in the phi network.
        phi_output_dim: Output dimension of the phi network.
        rho_hidden_dim: Number of hidden units in the rho network.
        lr: Learning rate for the optimizer.
        lr_step_size: Step size for the learning rate scheduler.
        lr_gamma: Multiplicative factor for the learning rate scheduler.
        batch_size: Batch size for training.
        num_epochs: Number of epochs for training.
        device: Device to use for training.
        dtype: Data type to use for training.
        progress: Whether to display a progress bar during training. If a dictionary is
            provided, it is passed to `tqdm` as keyword arguments.
    """

    def __init__(
        self,
        data: Dataset,
        phi_hidden_dim: int,
        phi_output_dim: int,
        rho_hidden_dim: int,
        lr: float = 0.001,
        lr_step_size: int = 10,
        lr_gamma: float = 0.1,
        batch_size: int = 64,
        num_epochs: int = 20,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        progress: dict[str, Any] | bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.data = data
        self.predictor = DeepSet(
            input_dim=self.data.n_features,
            phi_hidden_dim=phi_hidden_dim,
            phi_output_dim=phi_output_dim,
            rho_hidden_dim=rho_hidden_dim,
        ).to(device=device, dtype=dtype)
        self.is_trained = False

        self.tqdm_args: dict[str, Any] = {
            "desc": f"{self.__class__.__name__}, training",
            "unit": "epoch",
        }
        # HACK: parse additional args for the progress bar if any (we probably want
        #  something better)
        if isinstance(progress, bool):
            self.tqdm_args.update({"disable": not progress})
        elif isinstance(progress, dict):
            self.tqdm_args.update(progress)
        else:
            raise TypeError(f"Invalid type for progress: {type(progress)}")

    def fit(self, samples: dict[Sample, float]) -> Self:
        """

        Args:
            samples: A collection of utility samples

        Returns:

        """
        if self.is_trained:
            self.predictor.reset_parameters()
            self.is_trained = False

        dataset = SetDatasetRaw(samples, self.data)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        pbar = trange(self.num_epochs, **self.tqdm_args)
        for _ in pbar:
            epoch_loss = 0.0
            for set_tensor, target in dataloader:
                set_tensor = set_tensor.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = self.predictor(set_tensor)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            pbar.set_postfix_str(f"Loss: {epoch_loss / len(dataloader):.4f}")

        self.is_trained = True
        return self

    def predict(self, samples: Collection[Sample]) -> NDArray:
        """

        Args:
            samples: A collection of samples to predict their utility values.

        Returns:
            An array of values of dimension (len(samples), 1) with the predicted utility
        """
        if not samples:
            raise ValueError("The samples collection is empty.")
        max_set_size = max(len(s.subset) for s in samples)
        # grab device and dtype
        template = next(self.predictor.parameters())
        set_tensor = template.new_zeros(
            (len(samples), max_set_size, self.data.n_features)
        )
        for i, sample in enumerate(samples):
            for j, idx in enumerate(sample.subset):
                set_tensor[i, j] = torch.tensor(
                    self.data.data().x[idx],
                    device=template.device,
                    dtype=template.dtype,
                )
        with torch.no_grad():
            prediction = self.predictor(set_tensor)

        return cast(NDArray, prediction.cpu().numpy())
