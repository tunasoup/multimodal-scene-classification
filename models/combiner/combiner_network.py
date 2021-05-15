"""
Contains the trainable sub-network of an ensemble classifier.
Handles calling the training and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import show

from utility.fusion_functions import (train_nn_combiner_model,
                                      test_nn_combiner)
from definitions import (MODELS_TRAIN_OUTPUTS_FILE, MODELS_VAL_OUTPUTS_FILE,
                         MODELS_TEST_OUTPUTS_FILE,
                         BEST_COMBINER_MODEL)
from utility.utilities import FusionData, Features


class CombinerModel(nn.Module):
    def __init__(self):
        super(CombinerModel, self).__init__()
        self.input_dim = 40     # Label count * base model count
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.model(x)
        return x


if __name__ == '__main__':
    run_train = True
    run_test = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_ensemble = CombinerModel()

    if run_train:
        optimizer = optim.Adam(model_ensemble.parameters(), lr=0.000020)
        epochs = 100
        batch_size = 32
        feature_train = Features(feature_file=MODELS_TRAIN_OUTPUTS_FILE,
                                 arg_names=['X', 'y'])
        data_train = FusionData(features=[feature_train], do_reshape=False)
        feature_val = Features(feature_file=MODELS_VAL_OUTPUTS_FILE,
                               arg_names=['X', 'y'])
        data_val = FusionData(features=[feature_val], do_reshape=False)
        train_nn_combiner_model(model=model_ensemble,
                                optimizer=optimizer,
                                train_data=data_train.get_data(),
                                val_data=data_val.get_data(),
                                best_model=BEST_COMBINER_MODEL,
                                device=device,
                                epochs=epochs,
                                batch_size=batch_size)

    if run_test:
        feature_test = Features(feature_file=MODELS_TEST_OUTPUTS_FILE,
                                arg_names=['X', 'y'])
        data_test = FusionData(features=[feature_test], do_reshape=False)
        model_ensemble.load_state_dict(torch.load(BEST_COMBINER_MODEL))
        model_ensemble.eval()
        test_nn_combiner(model=model_ensemble,
                         test_data=data_test.get_data(),
                         device=device,
                         verbose=True)

    show()

"""
with logits, lr=0.000015, test: 0.872 (epoch 87) batchnormed, 0.871 without
with probabilities, lr=0.000025, test: 0.872 (epoch 46) batchnormed, 0.849 without
with softmax of probabilities, lr=0.000025 test: 0.872 (epoch 46) batchnormed, 0.814 without higher lr
with softmax of logits, lr=0.000020, 0.865, (epoch 46) batchnormed,
with logits:
test: 0.872     val: 0.892     val_loss: 0.2937   avg: 86.61%	   test_loss: 0.3840
optimizer = optim.Adam(model_ensemble.parameters(), lr=0.000015)
epochs = 92 # (87), 0.871 without batchnorm
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.5),

            nn.Linear(32, 10)
        )
"""
