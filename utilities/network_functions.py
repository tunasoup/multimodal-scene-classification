"""
Contains functions needed for training and evaluating a unimodal classifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from definitions import SEED, LABEL_COUNT, EPOCHS_AFTER_NEW_BEST
from utilities.global_utilities import create_file_directory

torch.manual_seed(SEED)


class FeatureData(Dataset):
    def __init__(self, feature_file: str):
        """
        Generate a torch Dataset from feature vectors.

        Meant to be used for training a network.

        Args:
            feature_file: an npz file holding feature vectors with keywords
                          'X' and 'y'
        """
        self.data = np.load(feature_file)
        self.X = self.data['X']
        self.y = self.data['y']

        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X)

        if isinstance(self.y, np.ndarray):
            self.y = torch.from_numpy(self.y)
            self.y = self.y.type(dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx]
        label = self.y[idx]

        return sample, label


def train_nn(model: nn.Module, optimizer: optim.Optimizer, train_features: str,
             val_features: str, best_model: str,
             device: str, epochs: int, batch_size: int):
    """
    Train a NN. For every epoch train and validate with beforehand
    obtained feature files, and save the best model.

    Args:
        model: model to be trained
        optimizer: optimizer with set parameters
        train_features: a path to a training feature file containing feature vectors
        val_features: a path to a validation feature file containing feature vectors
        best_model: a path to a file which will contain the best performing model state
        device: device to run pytorch on
        epochs: maximum number of full training data iterations
        batch_size: number of training samples used before optimizing
    """
    create_file_directory(best_model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = DataLoader(FeatureData(train_features),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(FeatureData(val_features),
                            batch_size=batch_size)

    acc_of_best = 0.0
    lowest_loss = 100   # Random high loss

    train_loss = []
    val_loss = []
    best_epoch = 0

    print(f'Training for {epochs} epochs')
    print(model)

    for epoch in range(epochs):
        train_e_loss = train(loader=train_loader, model=model, criterion=criterion,
                             optimizer=optimizer, device=device, epoch=epoch)
        acc, val_e_loss = validate(loader=val_loader, model=model, criterion=criterion,
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


def train(loader: DataLoader, model: nn.Module, criterion: nn.Module,
          optimizer: optim.Optimizer, device: str, epoch: int) -> float:
    """
    Train a NN for one epoch.

    Args:
        loader: torch Dataloader with a training dataset
        model: the network model to be trained
        criterion: the loss function
        optimizer: the network's optimizer
        device: device to run pytorch on
        epoch: current epoch number (0-indexing)

    Returns:
        mean loss
    """

    correct = 0
    running_loss = 0
    n = len(loader.dataset)

    model.train()
    batch_idx = 0
    for batch_idx, data in enumerate(loader):

        inputs = data[0].contiguous()
        labels = data[1].contiguous()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.data.item()
        correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        optimizer.step()

    mean_loss = running_loss / (batch_idx+1)
    print(f'epoch {epoch + 1} training:\t\t\t'
          f'loss: {mean_loss:.4f},\t\t\t'
          f'accuracy: {correct / n:3f}')

    return mean_loss


def validate(loader: DataLoader, model: nn.Module, criterion: nn.Module,
             device: str, epoch: int) -> (float, float):
    """
    Validate a NN with the validation data.

    Args:
        loader: torch Dataloader with a validation dataset
        model: the model to be validated
        criterion: the loss function
        device: device to run pytorch on
        epoch: current epoch number (0-indexing)

    Returns:
        validation accuracy and mean
    """
    correct = 0
    val_loss = 0
    n = len(loader.dataset)

    model.eval()
    batch_idx = 0
    for batch_idx, data in enumerate(loader):
        inputs = data[0].contiguous()
        labels = data[1].contiguous()

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.data.item()
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    acc = correct / n
    mean_loss = val_loss / (batch_idx+1)
    print(f'epoch {epoch + 1} validation:\t\t'
          f'val_loss: {mean_loss:.4f},\t\t'
          f'val_accuracy: {acc:.3f}')
    return acc, mean_loss


def test_nn(model: nn.Module, test_features: str,
            device: str, verbose: bool = False,
            batch_size: int = 32,
            take_batch_mean: bool = False) -> (float, float, torch.Tensor):
    """
    Test a NN with a beforehand obtained feature file and set state.
    Calculates accuracy, mean, and predicted label counts. The statistics
    can be calculated for either every feature vector, or on a file level with
    a mean of corresponding vector outputs.

    Args:
        model: model to be tested
        test_features: path to a testing feature file containing feature vectors
        device: device to run pytorch on
        verbose: print more information if true
        batch_size: batch size of the dataloader
        take_batch_mean: if true, take the mean of a batch's results.
                         Note that then batch_size should equal the amount of
                         feature vectors per file

    Returns:
        accuracy, mean loss, predicted labels
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()

    if verbose:
        print("Testing model ...")

    test_loader = DataLoader(FeatureData(test_features),
                             batch_size=batch_size)

    n = len(test_loader.dataset)
    if take_batch_mean:
        n = n // batch_size

    preds = torch.zeros(LABEL_COUNT, dtype=torch.long).to(device)
    correct = 0
    test_loss = 0.0

    batch_idx = 0
    for batch_idx, data in enumerate(test_loader):
        inputs = data[0].contiguous()
        labels = data[1].contiguous()

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.data.item()
            if take_batch_mean:     # combine file segments
                outputs = torch.mean(outputs, dim=0)
                outputs = torch.argmax(outputs)
                correct += torch.sum(outputs == labels[0]).item()
                preds[outputs] += 1
            else:                   # test segments individually
                outputs = torch.argmax(outputs, dim=1)
                correct += torch.sum(outputs == labels).item()
                counts = torch.bincount(outputs)
                preds[0:len(counts)] += counts

    acc = correct / n
    mean_loss = test_loss / (batch_idx+1)

    if verbose:
        print(f'test_loss: {mean_loss:.4f},\t\t'
              f'test_accuracy: {acc:.3f}')

    return acc, mean_loss, preds
