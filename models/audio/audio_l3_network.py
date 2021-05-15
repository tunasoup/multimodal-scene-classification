"""
Contains the trainable sub-network of an audio classifier based on the L3
pre-trained feature extractor. Handles calling the training and evaluation.
"""
from matplotlib.pyplot import show
import torch
import torch.nn as nn
import torch.optim as optim

from utility.network_functions import train_nn, test_nn
from definitions import (AUDIO_L3_TRAIN_FEATURES_FILE,
                         AUDIO_L3_VAL_FEATURES_FILE,
                         AUDIO_L3_TEST_FEATURES_FILE, BEST_AUDIO_L3_MODEL,
                         VECTORS_PER_AUDIO, AUDIO_L3_EMBED)


class AudioL3Model(nn.Module):
    def __init__(self):
        super(AudioL3Model, self).__init__()
        self.input_dim = AUDIO_L3_EMBED
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
    run_test = False               # Test on segment level
    run_mean_test = True           # Test on file level

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = AudioL3Model()

    if run_train:
        optimizer = optim.Adam(model.parameters(), lr=0.0000085)
        epochs = 100
        batch_size = 128
        train_nn(model=model, optimizer=optimizer,
                 train_features=AUDIO_L3_TRAIN_FEATURES_FILE,
                 val_features=AUDIO_L3_VAL_FEATURES_FILE,
                 best_model=BEST_AUDIO_L3_MODEL,
                 device=device,
                 epochs=epochs, batch_size=batch_size)

    if run_test:
        model.load_state_dict(torch.load(BEST_AUDIO_L3_MODEL))
        model.eval()
        test_nn(model=model,
                test_features=AUDIO_L3_TEST_FEATURES_FILE,
                device=device,
                verbose=True)

    if run_mean_test:
        model.load_state_dict(torch.load(BEST_AUDIO_L3_MODEL))
        model.eval()
        test_nn(model=model,
                test_features=AUDIO_L3_TEST_FEATURES_FILE,
                device=device,
                verbose=True,
                batch_size=VECTORS_PER_AUDIO,
                take_batch_mean=True)

    show()

"""
test: 0.684    val: 0.578    val_loss: 1.1170    avg: 68.65%     test_loss: 1.0586
optimizer = optim.Adam(model.parameters(), lr=0.0000085)
epochs = 100    # (47)
batch_size = 128
test: 0.681    val: 0.576    val_loss: 1.1184    avg: 68.29%     test_loss: 1.0630
optimizer = optim.Adam(model.parameters(), lr=0.000020)
epochs = 30 #(9) overfits
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

test: 0.674    val: 0.585    val_loss: 1.0935    avg: 67.30%     test_loss: 1.0414
optimizer = optim.Adam(model.parameters(), lr=0.000020)
epochs = 40  # (37)
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 10)
        )
"""