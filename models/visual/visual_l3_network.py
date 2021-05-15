"""
Contains a trainable part of a visual classifier based on the L3
pre-trained feature extractor. Handles calling the training and evaluation.
"""
from matplotlib.pyplot import show
import torch
import torch.nn as nn
import torch.optim as optim

from utility.network_functions import train_nn, test_nn
from definitions import (VISUAL_L3_TRAIN_FEATURES_FILE,
                         VISUAL_L3_VAL_FEATURES_FILE,
                         VISUAL_L3_TEST_FEATURES_FILE, BEST_VISUAL_L3_MODEL,
                         FRAMES_PER_VIDEO, VISUAL_L3_EMBED)


class VisualL3Model(nn.Module):
    def __init__(self):
        super(VisualL3Model, self).__init__()
        self.input_dim = VISUAL_L3_EMBED
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 10)
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

    model = VisualL3Model()

    if run_train:
        optimizer = optim.Adam(model.parameters(), lr=0.000025)
        epochs = 90
        batch_size = 32
        train_nn(model=model, optimizer=optimizer,
                 train_features=VISUAL_L3_TRAIN_FEATURES_FILE,
                 val_features=VISUAL_L3_VAL_FEATURES_FILE,
                 best_model=BEST_VISUAL_L3_MODEL,
                 device=device,
                 epochs=epochs, batch_size=batch_size)

    if run_test:
        model.load_state_dict(torch.load(BEST_VISUAL_L3_MODEL))
        model.eval()
        test_nn(model=model,
                test_features=VISUAL_L3_TEST_FEATURES_FILE,
                device=device,
                verbose=True)

    if run_mean_test:
        model.load_state_dict(torch.load(BEST_VISUAL_L3_MODEL))
        model.eval()
        test_nn(model=model,
                test_features=VISUAL_L3_TEST_FEATURES_FILE,
                device=device,
                verbose=True,
                batch_size=FRAMES_PER_VIDEO,
                take_batch_mean=True)

    show()

"""
test: 0.658     val: 0.719      val_loss: 0.7440    avg: 64.29%     test_loss: 0.9045
optimizer = optim.Adam(model.parameters(), lr=0.000025)
epochs = 90 #(86)
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 10)
        )
"""
