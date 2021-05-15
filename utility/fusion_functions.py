"""
Contains functions for late fusion methods, including the functions
required by a NN combiner.
"""
from typing import Callable, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from definitions import LABEL_COUNT, SEED, EPOCHS_AFTER_NEW_BEST
from utility.global_utilities import create_file_directory

torch.manual_seed(SEED)


class FeatureDataCollection(Dataset):
    def __init__(self, fusion_data: List[Union[np.ndarray, torch.Tensor]]):
        """
        Generate a torch dataset from a list of arrays or tensors.

        Args:
            fusion_data: a list of arrays or tensors, consisting of features or
                         base model outputs. Final element contains ground truth labels.
        """
        self.all_data = fusion_data

        # Convert numpy arrays to tensors
        for idx, samples in enumerate(self.all_data):

            if isinstance(samples, np.ndarray):
                tensor_samples = torch.from_numpy(samples)

                if idx == len(self.all_data) - 1:   # Change labels to long
                    tensor_samples = tensor_samples.type(dtype=torch.long)

                self.all_data[idx] = tensor_samples

    def __len__(self):
        return len(self.all_data[-1])

    def __getitem__(self, idx) -> List[Union[np.ndarray, torch.Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = [a[idx] for a in self.all_data]

        return data


def batch_prediction_processing(predictions: torch.Tensor) -> torch.Tensor:
    """
    Change the prediction type from logit to (softmaxed) probabilities.
    Requires commenting out code for different types.

    Combiner model requires a training session for each separate type.
    Only one type supported at a time.

    Args:
        predictions: outputs of the base models

    Returns:
        a prediction tensor
    """
    #predictions = inverse_logit(predictions)
    #predictions = softmax(predictions, dim=2)
    return predictions


def inverse_logit(predictions: torch.Tensor) -> torch.Tensor:
    """
    Take the inverse of logits.

    Args:
        predictions: outputs of the base models

    Returns:
        a probability tensor
    """
    expo = torch.exp(predictions)
    inputs = expo / (expo + 1)
    return inputs


def train_nn_combiner_model(model: nn.Module,
                            optimizer: optim.Optimizer, train_data: List[np.ndarray],
                            val_data: List[np.ndarray], best_model: str,
                            device: str, epochs: int, batch_size: int):
    """
    Train a NN based combiner with the given arguments.

    Args:
        model: the neural network combiner model to be trained
        optimizer: optimizer with set parameters
        train_data: a list of arrays with base model outputs on training data
        val_data: a list of arrays with base model outputs on validation data
        best_model: a path to a file which will contain the best performing model state
        device: device to run pytorch on
        epochs: maximum number of full training data iterations
        batch_size: number of training samples used before optimizing
    """
    create_file_directory(best_model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = DataLoader(FeatureDataCollection(train_data),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(FeatureDataCollection(val_data),
                            batch_size=batch_size)

    acc_of_best = 0.0
    lowest_loss = 100  # Random high loss

    train_loss = []
    val_loss = []
    best_epoch = 0

    print(f'Training for {epochs} epochs')
    print(model)

    for epoch in range(epochs):
        train_e_loss = train_nn_combiner(loader=train_loader, model=model,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         device=device, epoch=epoch)
        acc, val_e_loss = validate_nn_combiner(loader=val_loader,
                                               model=model,
                                               criterion=criterion,
                                               device=device, epoch=epoch)

        train_loss.append(train_e_loss)
        val_loss.append(val_e_loss)

        # End training if validation loss has not improved in a while
        if len(val_loss) - np.argmin(val_loss) > EPOCHS_AFTER_NEW_BEST:
            break

        # Save best state
        if val_e_loss < lowest_loss:
            lowest_loss = val_e_loss
            acc_of_best = acc
            best_epoch = epoch+1
            torch.save(model.state_dict(), best_model)

    print(f'Finished training, lowest val_loss {lowest_loss:.4f}, '
          f'with val_acc: {acc_of_best:.3f}, on epoch {best_epoch}\n')

    # Plot loss graph
    x_values = np.arange(1, len(train_loss)+1)
    plt.plot(x_values, train_loss)
    plt.plot(x_values, val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'])
    plt.draw()


def train_nn_combiner(loader: DataLoader, model: nn.Module, criterion: nn.Module,
                      optimizer: optim.Optimizer, device: str,
                      epoch: int) -> float:
    """
    Train a NN based combiner for one epoch.

    Args:
        loader: torch Dataloader with a training dataset
        model: the neural network combiner model
        criterion: the loss function
        optimizer: the optimizer with set parameters
        device: device to run pytorch on
        epoch: the current epoch (0-indexing)

    Returns:
        the mean loss of the model for training data
    """
    correct = 0
    running_loss = 0
    n = len(loader.dataset)

    model.train()
    batch_idx = 0
    for batch_idx, data in enumerate(loader):
        batch_samples = data[0].contiguous().to(device)
        batch_samples = batch_prediction_processing(batch_samples)

        labels = data[-1].contiguous()
        labels = labels.to(device)

        mini_length = len(labels)
        model_count = batch_samples.shape[1]
        inputs = torch.empty((mini_length, model_count, LABEL_COUNT), device=device)
        for mini_idx in range(mini_length):
            samples = batch_samples[mini_idx]
            inputs[mini_idx] = samples

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.data.item()
        correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        optimizer.step()

    mean_loss = running_loss / (batch_idx + 1)
    print(f'epoch {epoch + 1} training:\t\t\t'
          f'loss: {mean_loss:.4f},\t\t\t'
          f'accuracy: {correct / n:3f}')

    return mean_loss


def validate_nn_combiner(loader: DataLoader, model: nn.Module,
                         criterion: nn.Module,
                         device: str, epoch: int) -> (float, float):
    """
    Validate a NN based combiner with the validation data.

    Args:
        loader: torch Dataloader with a validation dataset
        model: the combiner neural network model
        criterion: the loss function
        device: device to run pytorch on
        epoch: the current epoch (0-indexing)

    Returns:
        the accuracy and mean loss of the model for validation data
    """
    correct = 0
    running_loss = 0
    n = len(loader.dataset)

    model.eval()
    batch_idx = 0
    for batch_idx, data in enumerate(loader):
        batch_samples = data[0].contiguous().to(device)
        batch_samples = batch_prediction_processing(batch_samples)

        labels = data[-1].contiguous()
        labels = labels.to(device)

        mini_length = len(labels)
        model_count = batch_samples.shape[1]
        inputs = torch.empty((mini_length, model_count, LABEL_COUNT), device=device)
        with torch.no_grad():
            for mini_idx in range(mini_length):
                samples = batch_samples[mini_idx]
                inputs[mini_idx] = samples

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.data.item()
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    acc = correct / n
    mean_loss = running_loss / (batch_idx + 1)
    print(f'epoch {epoch + 1} validation:\t\t\t'
          f'val_loss: {mean_loss:.4f},\t\t\t'
          f'val_accuracy: {acc:3f}')

    return acc, mean_loss


def test_nn_combiner(model: nn.Module,
                     test_data: List[np.ndarray],
                     device: str, verbose: bool = False) -> (float, float):
    """
    Test a NN based combiner with test data.

    Args:
        model: the neural network combiner model
        test_data: a list of arrays with base model outputs on test data
        device: device to run pytorch on
        verbose: print more information if true

    Returns:
        the accuracy and mean loss of the model
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if verbose:
        print("Testing model ...")

    loader = DataLoader(FeatureDataCollection(test_data),
                        batch_size=32)

    correct = 0
    running_loss = 0
    n = len(loader.dataset)

    model.eval()
    batch_idx = 0
    for batch_idx, data in enumerate(loader):
        batch_samples = data[0].contiguous().to(device)
        batch_samples = batch_prediction_processing(batch_samples)

        labels = data[-1].contiguous()
        labels = labels.to(device)

        mini_length = len(labels)
        model_count = batch_samples.shape[1]
        inputs = torch.empty((mini_length, model_count, LABEL_COUNT), device=device)
        with torch.no_grad():
            for mini_idx in range(mini_length):
                samples = batch_samples[mini_idx]
                inputs[mini_idx] = samples

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.data.item()
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    acc = correct / n
    mean_loss = running_loss / (batch_idx + 1)

    if verbose:
        print(f'test_loss: {mean_loss:.4f},\t\t'
              f'test_accuracy: {acc:.3f}')

    return acc, mean_loss


def test_single_model(model: nn.Module, all_data: List[torch.Tensor],
                      device: str) -> (float, torch.Tensor):
    """
    Count the accuracy with the given data for a system that fuses the
    probabilities of two classifiers. Also count the prediction labels.

    Args:
        model: base model to be tested
        all_data: a list of tensors with first tensor containing features and
                  the other ground truth labels
        device: device to run pytorch on

    Returns:
        model's accuracy and predicted labels
    """
    test_loader = DataLoader(FeatureDataCollection(all_data),
                             batch_size=32)

    preds = torch.zeros(LABEL_COUNT, dtype=torch.long).to(device)
    correct = 0
    n = len(test_loader.dataset)

    for batch_idx, data in enumerate(test_loader):
        samples = data[0].contiguous()
        samples = samples.to(device)

        labels = data[-1].contiguous()
        labels = labels.to(device)

        mini_length = len(labels)
        with torch.no_grad():
            for mini_idx in range(mini_length):
                model_output = model(samples[mini_idx])
                model_output = torch.mean(model_output, dim=0)
                outputs = model_output.squeeze(dim=0)
                output = torch.argmax(outputs, dim=0)
                correct += (output == labels[mini_idx]).item()
                preds[output] += 1

    acc = correct / n
    return acc, preds


def save_model_outputs(models: List[nn.Module],
                       all_data: List[np.ndarray],
                       device: str,
                       file_out_path: str):
    """
    Save the logit tensors of the given models to a file.

    Meant to be used after training all the classifiers for quick access
    for the combiners.

    Args:
        models: loaded base models
        all_data: a list of arrays with feature data, where the final array contains labels
        device: device to run pytorch on
        file_out_path: a path to a file where the outputs and true labels are saved
    """
    create_file_directory(file_out_path)

    loader = DataLoader(FeatureDataCollection(all_data),
                        batch_size=32)

    n = len(loader.dataset)
    print(f'Saving outputs for {n} files ...')

    all_outputs = torch.empty((n, len(models), LABEL_COUNT), device=device)
    counter = 0

    inputs: Dict[int, torch.Tensor] = {}
    for batch_idx, data in enumerate(loader):
        for i in range(len(models)):
            samples = data[i].contiguous()
            samples = samples.to(device)
            inputs[i] = samples

        labels = data[-1].contiguous()
        labels = labels.to(device)

        mini_length = len(labels)
        with torch.no_grad():
            for mini_idx in range(mini_length):

                outputs = torch.empty(len(models), LABEL_COUNT).to(device)
                for i in range(len(models)):
                    model = models[i]
                    model_output = model(inputs[i][mini_idx])
                    model_output = torch.mean(model_output, dim=0)
                    outputs[i] = model_output

                all_outputs[counter] = outputs
                counter += 1

    np.savez_compressed(file_out_path,
                        X=all_outputs.detach().cpu().numpy(),
                        y=all_data[-1])


def test_combiner(output_data: List[torch.Tensor],
                  device: str,
                  rule_function: Callable,
                  weights: torch.Tensor = None,
                  fusion_model: nn.Module = None) -> (float, torch.Tensor):
    """
    Test a combiner with the outputs of the models.

    Args:
        output_data: model outputs as a list of tensors, with final tensor being labels
        device: device to run pytorch on
        rule_function: a function with a combination rule
        weights: weights for classifiers or their predictions, dimensionality
                 depends on the rule function
        fusion_model: a model that is used with the deep rule function

    Returns:
        the accuracy and the predictions of the combiner
    """

    test_loader = DataLoader(FeatureDataCollection(output_data),
                             batch_size=32)

    preds = torch.zeros(LABEL_COUNT, dtype=torch.long).to(device)
    correct = 0
    n = len(test_loader.dataset)

    for batch_idx, data in enumerate(test_loader):
        batch_outputs = data[0].contiguous().to(device)
        batch_outputs = batch_prediction_processing(batch_outputs)

        labels = data[-1].contiguous()
        labels = labels.to(device)

        mini_length = len(labels)
        for mini_idx in range(mini_length):
            outputs = batch_outputs[mini_idx]
            outputs = rule_function(outputs, weights, fusion_model).to(device)

            output = torch.argmax(outputs, dim=0)
            correct += (output == labels[mini_idx]).item()
            preds[output] += 1

    acc = correct / n
    return acc, preds


def deep_rule_func(predictions: torch.Tensor, _, model: nn.Module) -> torch.Tensor:
    """
    Use a trained neural network model for the predictions.

    Args:
        predictions: outputs of the base models
        _:
        model: a trained model

    Returns:
        a prediction tensor
    """
    inputs = torch.reshape(predictions, (1, 40))
    return model(inputs).squeeze(dim=0)


def mean_rule_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Mean the predictions of different classifier outputs.

    Args:
        predictions: outputs of the base models

    Returns:
        a prediction tensor
    """
    return torch.mean(predictions, dim=0)


def weighted_mean_rule_func(predictions: torch.Tensor,
                            weights: torch.Tensor, *_) -> torch.Tensor:
    """
    Mean the predictions of different classifier outputs with classifier weights.

    Args:
        predictions: outputs of the base models
        weights: a one-dimensional tensor with a float for each classifier

    Returns:
        a prediction tensor
    """
    return torch.mean(predictions * weights.unsqueeze(dim=1), dim=0)


def weighted_label_mean_rule_func(predictions: torch.Tensor,
                                  weights: torch.Tensor, *_) -> torch.Tensor:
    """
    Mean the predictions of different classifier outputs with label weights.

    Args:
        predictions: outputs of the base models
        weights: a tensor with dimensions classifier count and label count, with
                 float for every prediction of a classifier

    Returns:
        a prediction tensor
    """
    return torch.mean(predictions * weights, dim=0)


def mode_rule_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Take the mode (most frequent) label of the predictions.
    On ties, torch prefers the smallest number.

    Args:
        predictions: outputs of the base models
        *_:

    Returns:
        a one-hot prediction tensor
    """
    inputs = torch.max(predictions, dim=1).indices
    mode_output = torch.mode(inputs, dim=0).values
    return torch.eye(LABEL_COUNT, device=predictions.device)[mode_output]


def custom_mode_rule_func(predictions: torch.Tensor,
                          weights: torch.Tensor, *_) -> torch.Tensor:
    """
    Take the mode (most frequent) label of the predictions.
    On ties, use the given weights to determine which classifier output that
    is found in the modes is used.

    Args:
        predictions: outputs of the base models
        weights: a one-dimensional tensor with classifier indices in the order
                 preference, used in tie type of situations
        *_:

    Returns:
        a one-hot prediction tensor
    """
    inputs = torch.max(predictions, dim=1).indices
    counts = torch.bincount(inputs)
    modes = torch.nonzero((counts == torch.max(counts, dim=0)[0])).squeeze(dim=1)
    if len(modes) > 1:
        for weight in weights:
            if inputs[weight.item()] in modes:
                modes = inputs[weight.item()]
                break
    return torch.eye(LABEL_COUNT, device=predictions.device)[modes.item()]


def median_rule_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Take the median (sorted middle value) label of the predictions.
    On even situation, torch prefers the smallest number.

    Args:
        predictions: outputs of the base models

    Returns:
        a one-hot prediction tensor
    """
    inputs = torch.max(predictions, dim=1).indices
    mode_output = torch.median(inputs, dim=0).values
    return torch.eye(LABEL_COUNT, device=predictions.device)[mode_output]


def custom_median_rule_func(predictions: torch.Tensor,
                            weights: torch.Tensor, *_) -> torch.Tensor:
    """
    Take the median (sorted middle value) label of the predictions.
    On even situation, select the middle-ish value which appears first
    in the priority weight tensor.

    Args:
        predictions: outputs of the base models
        weights: a one-dimensional tensor with classifier indices in the order
                 preference, used in tie type of situations
        *_:

    Returns:
        a one-hot prediction tensor
    """
    inputs = torch.max(predictions, dim=1).indices
    values, indices = torch.sort(inputs)
    length = len(values)
    mid_idx = length // 2
    median = values[mid_idx]
    if length % 2 == 0:
        for weight in weights:
            if inputs[weight.item()] in values[mid_idx-1:mid_idx+1]:
                median = inputs[weight.item()]
                break

    return torch.eye(LABEL_COUNT, device=predictions.device)[median]


def max_rule_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Combine the highest predictions of each label to one tensor.

    Args:
        predictions: outputs of the base models

    Returns:
        a prediction tensor
    """
    inputs = torch.max(predictions, dim=0)
    return inputs.values


def min_rule_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Combine the lowest predictions of each label to one tensor.

    Args:
        predictions: outputs of the base models

    Returns:
        a prediction tensor
    """
    inputs = torch.min(predictions, dim=0)
    return inputs.values


def product_rule_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Combine the lowest predictions of each label to one tensor.

    Args:
        predictions: outputs of the base models

    Returns:
        a prediction tensor
    """
    return torch.prod(predictions, dim=0)


def jury_func(predictions: torch.Tensor, *_) -> torch.Tensor:
    """
    Remove the highest and lowest predictions outputs of classifiers and
    take a mean of the remaining ones.

    Args:
        predictions: outputs of the base models
        *_:

    Returns:
        a prediction tensor
    """
    inputs = torch.sort(predictions.T, dim=1).values
    inputs = inputs[:, 1:-1].T
    return torch.sum(inputs, dim=0)


def weighted_jury_func(predictions: torch.Tensor,
                       weights: torch.Tensor, *_) -> torch.Tensor:
    """
    Remove the highest and lowest predictions outputs of classifiers and
    take a mean of the remaining ones. The predictions are first weighted.

    Args:
        predictions: outputs of the base models
        weights: a one-dimensional tensor with a float for each classifier
        *_:

    Returns:
        a prediction tensor
    """
    inputs = predictions * weights.unsqueeze(dim=1)
    inputs = torch.sort(inputs.T, dim=1).values
    inputs = inputs[:, 1:-1].T
    return torch.sum(inputs, dim=0)
