"""
Responsible for initiating the calculations and showing the performance of
different models, including the base models and the ensemble models constisting
of the base models and a combiner.
"""
from typing import Dict, Any, List, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import show

from models import (AudioVGGishModel, AudioL3Model, VisualResnetModel,
                    VisualL3Model, CombinerModel)
from utility.utilities import (FusionInfo, Features, FusionData,
                               create_model_and_features, load_model)
from utility.fusion_functions import (test_single_model,
                                      weighted_mean_rule_func, max_rule_func,
                                      product_rule_func,
                                      min_rule_func, jury_func,
                                      mode_rule_func, median_rule_func, deep_rule_func,
                                      test_combiner,
                                      custom_mode_rule_func, custom_median_rule_func,
                                      mean_rule_func, weighted_label_mean_rule_func)
from definitions import (BEST_AUDIO_VGGISH_MODEL,
                         BEST_VISUAL_RESNET_MODEL,
                         VISUAL_RESNET_TEST_FEATURES_FILE,
                         AUDIO_VGGISH_TEST_FEATURES_FILE,
                         LABELS, FRAMES_PER_VIDEO, VECTORS_PER_AUDIO,
                         BEST_VISUAL_L3_MODEL, BEST_AUDIO_L3_MODEL,
                         VISUAL_L3_TEST_FEATURES_FILE,
                         AUDIO_L3_TEST_FEATURES_FILE, VISUAL_RESNET_EMBED,
                         AUDIO_L3_EMBED, AUDIO_VGGISH_EMBED, VISUAL_L3_EMBED,
                         BEST_COMBINER_MODEL, MODELS_TEST_OUTPUTS_FILE)


plt.rc('font', size=18)     # Size for confusion matrix numbers


def calculate_model_statistics(fusions: Dict[str, FusionInfo],
                               device: str):
    """
    Calculate and print the statistics for the given models.
    The fusions consists of a combination method with data of multiple base
    models. The function allows having a single base model with feature data
    as a fusino as well.

    Args:
        fusions: a dict with fusion name as the key and FusionInfo as value
        device: device to run pytorch on
    """
    print(f'Calculating outputs for {len(fusions)} fusion(s) ...')
    scene_accuracies: Dict[str, Dict[str, float]] = {}
    scene_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
    for scene in LABELS:
        scene_accuracies[scene] = {}
        scene_predictions[scene] = {}

    # Loop every model
    for fusion_idx, fusion_name in enumerate(fusions.keys()):
        info = fusions[fusion_name]

        y = info.all_data.get_labels()
        # Go through all the labels
        for scene in LABELS.keys():
            i = LABELS[scene]
            b = y == i  # boolean index selector
            labels = y[b]
            acc = 0.0
            preds = 0

            # if occurences of the scene exists in the test data
            if labels.size > 0:

                # all scene data with labels
                data = [samples[b] for samples in info.all_data.get_data()]

                # single models
                if len(info.models) == 1:
                    acc, preds = test_single_model(model=info.models[0],
                                                   all_data=data,
                                                   device=device)

                # fusions
                elif len(info.models) > 1:
                    acc, preds = test_combiner(output_data=data,
                                               device=device,
                                               rule_function=info.rule_function,
                                               weights=info.weights,
                                               fusion_model=info.fusion_model)

            scene_accuracies[scene][fusion_name] = acc
            scene_predictions[scene][fusion_name] = preds

        print(f'{fusion_idx + 1} fusions(s) calculated')

    print_accuracies(scene_accuracies)

    # Calculate f1 scores and confusion matrices
    f1_scene_scores: Dict[str, Dict[str, float]] = {}
    f1_avg_scores: Dict[str, Dict[str, float]] = {}
    for fusion_name, info in fusions.items():

        predictions = {}
        for scene, scene_dict in scene_predictions.items():
            predictions[scene] = scene_dict[fusion_name]

        #print_predictions(predictions)

        conf_matrix = torch.vstack(list(predictions.values())).detach().cpu().numpy()

        f1_scene_scores[fusion_name] = calculate_f1_scene_scores(conf_matrix)
        f1_avg_scores[fusion_name] = calculate_avg_f1_scores(conf_matrix)

        if info.plot_cm:
            plot_confusion_matrix(conf_matrix)

    f1_scores = {}
    f1_scores = add_to_dict_with_swapped_keys(f1_scene_scores, f1_scores)
    f1_scores = add_to_dict_with_swapped_keys(f1_avg_scores, f1_scores)

    print_f1_scores(f1_scores)


def add_to_dict_with_swapped_keys(old_dict: Dict[Any, Dict],
                                  new_dict: Dict[Any, Dict]) -> Dict[Any, Dict]:
    """
    Swap the keys of two nested dictionaries in a new dictionary.
    {'Key1': {'Key2': 42}} -> {'Key2': {'Key1': 42}}
    Args:
        old_dict: a nested dictionary whose keys are to be swapped
        new_dict: an initiated dictionary, does not have to be empty

    Returns:
        the new_dict with the addition of swapped keys of the old_dict
    """
    for key1, nested_dict in old_dict.items():
        for key2, value in nested_dict.items():
            new_dict.setdefault(key2, {})[key1] = value

    return new_dict


def print_accuracies(scene_accuracies: Dict[str, Dict[str, float]]):
    """
    Print the accuracies of different scenes and the average accuracy of them
    for every fusion.
    Can also print a LaTeX friendly table elements.

    Args:
        scene_accuracies: a dict with a scene as a key and another dict as a
                          values. It has a fusion name as a key, and float
                          accuracy as value.
    """
    # Print scene accuracies
    latex_table = '\n\\hline'
    print('')
    for scene in scene_accuracies:
        table = f'\nAccuracies for {scene}:\n'
        latex_table += f'\n{scene} & '

        scene_acc = scene_accuracies[scene]
        for fusion_name in scene_acc:
            table += f'{fusion_name}: {100 * scene_acc[fusion_name]:.1f}%\t\t'
            latex_table += f'{100 * scene_acc[fusion_name]:.1f} & '

        latex_table = latex_table[:-2] + '\\\\'

        print(table)
    latex_table += '\n\\hline\n'

    # Print average accuracies
    fusion_names = []
    for scene in scene_accuracies.values():
        fusion_names = list(scene.keys())
        break

    table = f'\nTotal averages:\n'
    latex_table += 'average &'
    for fusion_name in fusion_names:
        accs = np.empty(len(scene_accuracies))

        for idx, scene_dict in enumerate(scene_accuracies.values()):
            accs[idx] = scene_dict[fusion_name]

        mean_acc = np.mean(accs)
        table += f'{fusion_name}: {100 * mean_acc:.1f}%\t\t'
        latex_table += f'{100 * mean_acc:.1f} & '

    latex_table = latex_table[:-2] + '\\\\\n\\hline'

    print(table)
    print(latex_table)


def calculate_f1_scene_scores(cm: np.ndarray) -> Dict[str, float]:
    """
    Calculate the f1 score for every scene.

    Args:
        cm: a confusion matrix made out of a numpy array

    Returns:
        dict with scene as key and score as value
    """
    label_to_scene = {v: k for k, v in LABELS.items()}
    scores = {}
    cm_t = cm.T
    for i in range(len(cm)):
        f1_scene_score = 2 * cm[i][i] / (np.sum(cm[i]) + np.sum(cm_t[i]))
        scores[label_to_scene[i]] = f1_scene_score

    return scores


def calculate_avg_f1_scores(cm: np.ndarray) -> Dict[str, float]:
    """
    Calculate the average f1 score.

    Args:
        cm: a confusion matrix made out of a numpy array

    Returns:
        dict with average method as key and score as value
    """
    scores = {}
    gt_cm = np.sum(cm, axis=1)
    y_true = np.repeat(np.arange(gt_cm.size), gt_cm)
    y_pred = []
    for row in cm:
        for idx, count in enumerate(row):
            if count > 0:
                y_pred.append(np.repeat(idx, count))
    y_pred = np.concatenate(y_pred).ravel()
    scores['macro'] = f1_score(y_true, y_pred, average='macro')
    scores['micro'] = f1_score(y_true, y_pred, average='micro')
    return scores


def print_f1_scores(f1_scores: Dict[str, Dict[str, float]]):
    """
    Print the fusion method f1 scores for every scene and their averages.

    Args:
        f1_scores: a dict with scene/average method as key and a dict with value,
        which consists of fusion method key and score value.
    """
    latex_table = '\n\\hline'
    print('')
    for case, fusions_score in f1_scores.items():
        table = f'\nF1 scores for {case}:\n'
        latex_table += f'\n{case} & '

        for fusion_name, score in fusions_score.items():
            table += f'{fusion_name}: {score:.3f}\t\t'
            latex_table += f'{score:.3f} & '

        latex_table = latex_table[:-2] + '\\\\'

        print(table)
    latex_table += '\n\\hline\n'
    print(latex_table)


def print_predictions(predictions: Dict[str, torch.Tensor]):
    """
    Print a fusion's predicted labels for every scene. Basically a poor version
    of a confusion matrix.
    Can also print a LaTeX friendly table elements.

    Args:
        predictions: a dict with scene as a key and a tensor of prediction
                     counts for every scene as a value.
    """
    label_to_scene = {v: k for k, v in LABELS.items()}
    latex_table = '\n\\hline'
    for scene in predictions:
        table = f'\nPredictions for {scene}:\n'
        latex_table += f'\n{scene} & '

        scene_occurences = predictions[scene]
        for label in range(len(scene_occurences)):
            table += f'{label_to_scene[label]}: {scene_occurences[label]}\t\t'
            latex_table += f'{scene_occurences[label]} & '

        latex_table = latex_table[:-2] + '\\\\'

        print(table)

    latex_table += '\n\\hline\n'
    #print(latex_table)


def plot_confusion_matrix(cm: np.ndarray):
    """
    Plot a confusion matrix from the predictions

    Args:
        cm: a confusion matrix made out of a numpy array
    """
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(colorbar=False, cmap='Blues', xticks_rotation=30, ax=ax)
    ax.set_xlabel('Predicted scene', fontsize=15)
    ax.set_xticklabels(list(LABELS.keys()), size=13, rotation=30, ha='right')
    ax.set_ylabel('True scene', fontsize=15)
    ax.set_yticklabels(list(LABELS.keys()), size=13)
    fig.subplots_adjust(bottom=0.15)


class InfoCreator:
    def __init__(self, models: List[nn.Module], data: FusionData):
        """
        Creates FusionInfo objects with the models and data given.

        Args:
            models: a list of base models
            data: a FusionData object
        """
        self.models = models
        self.data = data

    def create_info(self, rule_function: Callable, weights: torch.Tensor = None,
                    fusion_model: nn.Module = None, plot_cm: bool = False) -> FusionInfo:
        """
        Create a FusionInfo with the default models and data.

        Args:
            rule_function: a function with a combination rule
            weights: weights for classifiers or their predictions, dimensionality
                     depends on the rule function
            fusion_model: a model that is used with the deep rule function
            plot_cm: boolean for plotting the confusion matrix

        Returns:
            a FusionInfo object
        """
        info = FusionInfo(
            models=self.models,
            all_data=self.data,
            rule_function=rule_function,
            weights=weights,
            fusion_model=fusion_model,
            plot_cm=plot_cm
        )
        return info


if __name__ == '__main__':
    # load device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Initiate models and their feature data
    model_audio_vggish, feature_audio_vggish = create_model_and_features(
        model_class=AudioVGGishModel,
        saved_model=BEST_AUDIO_VGGISH_MODEL,
        feature_file=AUDIO_VGGISH_TEST_FEATURES_FILE,
        embed_size=AUDIO_VGGISH_EMBED,
        vector_count=VECTORS_PER_AUDIO,
        device=device
    )

    data_audio_vggish = FusionData(features=[feature_audio_vggish])
    info_audio_vggish = FusionInfo(models=[model_audio_vggish],
                                   all_data=data_audio_vggish,
                                   plot_cm=True)

    model_audio_l3, feature_audio_l3 = create_model_and_features(
        model_class=AudioL3Model,
        saved_model=BEST_AUDIO_L3_MODEL,
        feature_file=AUDIO_L3_TEST_FEATURES_FILE,
        embed_size=AUDIO_L3_EMBED,
        vector_count=VECTORS_PER_AUDIO,
        device=device)

    data_audio_l3 = FusionData(features=[feature_audio_l3])
    info_audio_l3 = FusionInfo(models=[model_audio_l3],
                               all_data=data_audio_l3,
                               plot_cm=True)

    model_visual_resnet, feature_visual_resnet = create_model_and_features(
        model_class=VisualResnetModel,
        saved_model=BEST_VISUAL_RESNET_MODEL,
        feature_file=VISUAL_RESNET_TEST_FEATURES_FILE,
        embed_size=VISUAL_RESNET_EMBED,
        vector_count=FRAMES_PER_VIDEO,
    device=device)

    data_visual_resnet = FusionData(features=[feature_visual_resnet])
    info_visual_resnet = FusionInfo(models=[model_visual_resnet],
                                    all_data=data_visual_resnet,
                                    plot_cm=True)

    model_visual_l3, feature_visual_l3 = create_model_and_features(
        model_class=VisualL3Model,
        saved_model=BEST_VISUAL_L3_MODEL,
        feature_file=VISUAL_L3_TEST_FEATURES_FILE,
        embed_size=VISUAL_L3_EMBED,
        vector_count=FRAMES_PER_VIDEO,
        device=device
    )

    data_visual_l3 = FusionData(features=[feature_visual_l3])
    info_visual_l3 = FusionInfo(models=[model_visual_l3],
                                all_data=data_visual_l3,
                                plot_cm=True)

    # Needs to be in the same order as they were saved in output_saving.py
    all_models = [model_audio_vggish, model_audio_l3,
                  model_visual_resnet, model_visual_l3]

    feature_all = Features(feature_file=MODELS_TEST_OUTPUTS_FILE)

    data_all = FusionData(features=[feature_all], do_reshape=False)

    ic = InfoCreator(models=all_models, data=data_all)

    # Create fusion infos
    info_mean = ic.create_info(mean_rule_func, plot_cm=True)
    info_min = ic.create_info(min_rule_func)
    info_max = ic.create_info(max_rule_func)
    info_jury = ic.create_info(jury_func)
    info_product = ic.create_info(product_rule_func)
    info_mode = ic.create_info(mode_rule_func)
    info_median = ic.create_info(median_rule_func)

    priority_weights = torch.tensor([2, 3, 1, 0], device=device)
    info_mode_priority = ic.create_info(custom_mode_rule_func,
                                        weights=priority_weights)
    info_median_priority = ic.create_info(custom_median_rule_func,
                                          weights=priority_weights)

    model_combiner = load_model(CombinerModel, BEST_COMBINER_MODEL, device)
    info_nn_combiner = ic.create_info(deep_rule_func,
                                      fusion_model=model_combiner)

    f1_label_weights = torch.tensor(
        [[0.374, 0.557, 0.519, 0.566, 0.888, 0.533, 0.599, 0.560, 0.838, 0.510],
         [0.702, 0.741, 0.565, 0.626, 0.869, 0.570, 0.673, 0.618, 0.821, 0.659],
         [0.694, 0.755, 0.881, 0.951, 0.894, 0.744, 0.846, 0.823, 0.809, 0.567],
         [0.345, 0.715, 0.743, 0.783, 0.848, 0.522, 0.582, 0.670, 0.740, 0.417]],
        device=device)
    f1_label_weights_norm = (f1_label_weights.T / torch.sum(f1_label_weights.T, dim=1, keepdims=True)).T

    f1_weights = torch.tensor([0.609, 0.684, 0.807, 0.658], device=device)
    f1_weights_norm = f1_weights / torch.sum(f1_weights)

    acc_weights = torch.tensor([0.598, 0.687, 0.798, 0.643], device=device)
    acc_weights_norm = acc_weights / torch.sum(acc_weights)

    info_acc_weighted_mean = ic.create_info(weighted_mean_rule_func,
                                            weights=acc_weights)

    info_f1_weighted_mean = ic.create_info(weighted_mean_rule_func,
                                           weights=f1_weights_norm)

    info_f1_weighted_label_mean = ic.create_info(weighted_label_mean_rule_func,
                                                 weights=f1_label_weights_norm)

    # Collect fusions for testing
    fusions = {
        #'audio_vggish': info_audio_vggish,
        #'audio_l3': info_audio_l3,
        #'visual_resnet': info_visual_resnet,
        #'visual_l3': info_visual_l3,
        'mean': info_mean,
        #'acc weighted mean': info_acc_weighted_mean,
        #'f1 weighted mean': info_f1_weighted_mean,
        'f1 weighted label mean': info_f1_weighted_label_mean,
        #'min': info_min,
        #'max': info_max,
        #'product': info_product,
        #'jury': info_jury,
        #'mode': info_mode,
        #'mode prio': info_mode_priority,
        #'median': info_median,
        #'median prio': info_median_priority,
        #'NN combiner': info_nn_combiner

    }

    calculate_model_statistics(fusions=fusions, device=device)
    show()
