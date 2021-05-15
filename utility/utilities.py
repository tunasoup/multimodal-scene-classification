"""
Contains utility functions.

Meant to be used with the main virtual environment.
"""
from typing import NamedTuple, List, Type, Callable, Tuple

import numpy as np
import torch.nn as nn
import torch


class Features(NamedTuple):
    """
    Holds information regarding features in a feature file
    """
    feature_file: str
    arg_names: List[str] = ['X', 'y']
    embed_size: int = 0
    vector_count: int = 0


class FusionData:
    def __init__(self, features: List[Features], do_reshape: bool = True):
        """
        Stacks the data samples referenced by the given Feature objects.
        The samples can either be features or base model outputs.
        Features can be reshaped according to their vector count
        for calculcating the mean of a single file's classifier outputs.

        Args:
            features: a list of Features
        """
        self.features = features
        self.all_data = []

        for idx, feature in enumerate(self.features):
            data = np.load(feature.feature_file)
            samples = data[feature.arg_names[0]]

            if do_reshape:
                samples = np.reshape(samples,
                                     (len(samples) // feature.vector_count,
                                      feature.vector_count, feature.embed_size))
            self.all_data.append(samples)

            if idx == len(features)-1:
                labels = data[feature.arg_names[1]]

                if do_reshape:
                    labels = labels[::feature.vector_count]

                self.all_data.append(labels)

    def get_data(self) -> List[np.ndarray]:
        """
        Returns:
            a list of arrays consisting of feature data and labels
        """
        return self.all_data

    def get_labels(self) -> np.ndarray:
        """
        Returns:
            an array of labels
        """
        return self.all_data[-1]


def load_model(model_class: Type[nn.Module], saved_model: str,
               device: str) -> nn.Module:
    """
    Load a model with the given state and set it to evalute.

    Args:
        model_class: the class of the model
        saved_model: a path to a file containing the saved model state
        device: device to run pytorch on

    Returns:
        a loaded model
    """
    model = model_class()
    model.load_state_dict(torch.load(saved_model))
    model.to(device)
    model.eval()

    return model


def create_model_and_features(model_class: Type[nn.Module], saved_model: str,
                              feature_file: str, embed_size: int,
                              vector_count: int, device: str) -> (nn.Module, Features):
    """
    Load a model and create a single Features object

    Args:
        model_class: the class of the model
        saved_model: a path to a file containing the saved model state
        feature_file: a path to a feature file containing feature fectors
        embed_size: size of a single feature embed
        vector_count: number of feature vectors per file
        device: device to run pytorch on

    Returns:
        a loaded model and a Features object
    """
    model = load_model(model_class, saved_model, device)
    feature = Features(feature_file=feature_file,
                       arg_names=['X', 'y'],
                       embed_size=embed_size,
                       vector_count=vector_count)

    return model, feature


def create_multiple_features(feature_files: List[str], embed_size: int,
                             vector_count: int) -> Tuple:
    """
    Create a Feature object from every given feature file.

    Args:
        feature_files: a list of paths to feature files containing feature vectors
        embed_size: size of a single feature embed
        vector_count: number of feature vectors per file

    Returns:
        a tuple of the created Feature objects
    """
    features = []
    for feature_file in feature_files:
        feature = Features(feature_file=feature_file,
                           arg_names=['X', 'y'],
                           embed_size=embed_size,
                           vector_count=vector_count)
        features.append(feature)

    return tuple(features)


class FusionInfo(NamedTuple):
    """
    Contains information of a fusion model required to test it.
    """
    models: List[nn.Module]
    all_data: FusionData
    rule_function: Callable = None  # Combiner rule in fusion
    weights: torch.Tensor = None    # Weights in a combiner
    fusion_model: nn.Module = None  # Model used by deep rule based combiner
    plot_cm: bool = False           # Plot confusion matrix
