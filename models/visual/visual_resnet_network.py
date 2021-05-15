"""
Contains a trainable part of a visual classifier based on the ResNet50
pre-trained feature extractor. Handles calling the training and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import show

from utility.network_functions import train_nn, test_nn
from definitions import (VISUAL_RESNET_TRAIN_FEATURES_FILE,
                         VISUAL_RESNET_VAL_FEATURES_FILE,
                         VISUAL_RESNET_TEST_FEATURES_FILE,
                         BEST_VISUAL_RESNET_MODEL,
                         FRAMES_PER_VIDEO, VISUAL_RESNET_EMBED)


class VisualResnetModel(nn.Module):
    def __init__(self):
        super(VisualResnetModel, self).__init__()
        self.input_dim = VISUAL_RESNET_EMBED
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    run_train = True
    run_test = False            # Test on segment level
    run_mean_test = True        # Test on file level

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = VisualResnetModel()

    if run_train:
        optimizer = optim.Adam(model.parameters(), lr=0.000025)
        epochs = 60
        batch_size = 32
        train_nn(model=model, optimizer=optimizer,
                 train_features=VISUAL_RESNET_TRAIN_FEATURES_FILE,
                 val_features=VISUAL_RESNET_VAL_FEATURES_FILE,
                 best_model=BEST_VISUAL_RESNET_MODEL,
                 device=device,
                 epochs=epochs, batch_size=batch_size)

    if run_test:
        model.load_state_dict(torch.load(BEST_VISUAL_RESNET_MODEL))
        model.eval()
        test_nn(model=model, test_features=VISUAL_RESNET_TEST_FEATURES_FILE,
                device=device,
                verbose=True)

    if run_mean_test:
        model.load_state_dict(torch.load(BEST_VISUAL_RESNET_MODEL))
        model.eval()
        test_nn(model=model, test_features=VISUAL_RESNET_TEST_FEATURES_FILE,
                device=device,
                verbose=True,
                batch_size=FRAMES_PER_VIDEO,
                take_batch_mean=True)

    show()

"""
test: 0.807     val: 0.820      val_loss: 0.4879    avg: 79.76%     test_loss: 0.6033
optimizer = optim.Adam(model.parameters(), lr=0.000025)
epochs = 70 #(26)
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

test: 0.796     val: 0.814      val_loss: 0.4512    avg: 79.09%     test_loss: 0.6099
optimizer = optim.Adam(model.parameters(), lr=0.000025)
epochs = 70 #(55)
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 10)
        )                     
"""
