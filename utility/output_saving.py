"""
The base model outputs for each data set (train, val, test) is saved.
"""
import torch

from models import AudioVGGishModel, AudioL3Model, VisualResnetModel, VisualL3Model
from definitions import (BEST_AUDIO_VGGISH_MODEL,
                         BEST_VISUAL_RESNET_MODEL,
                         VISUAL_RESNET_TEST_FEATURES_FILE,
                         AUDIO_VGGISH_TEST_FEATURES_FILE,
                         FRAMES_PER_VIDEO, VECTORS_PER_AUDIO,
                         BEST_VISUAL_L3_MODEL, BEST_AUDIO_L3_MODEL,
                         VISUAL_L3_TEST_FEATURES_FILE,
                         AUDIO_L3_TEST_FEATURES_FILE, VISUAL_RESNET_EMBED,
                         AUDIO_L3_EMBED, AUDIO_VGGISH_EMBED, VISUAL_L3_EMBED,
                         MODELS_TEST_OUTPUTS_FILE,
                         AUDIO_L3_TRAIN_FEATURES_FILE,
                         AUDIO_L3_VAL_FEATURES_FILE,
                         AUDIO_VGGISH_TRAIN_FEATURES_FILE,
                         AUDIO_VGGISH_VAL_FEATURES_FILE,
                         VISUAL_L3_TRAIN_FEATURES_FILE,
                         VISUAL_L3_VAL_FEATURES_FILE,
                         VISUAL_RESNET_TRAIN_FEATURES_FILE,
                         VISUAL_RESNET_VAL_FEATURES_FILE,
                         MODELS_TRAIN_OUTPUTS_FILE, MODELS_VAL_OUTPUTS_FILE)
from utility.fusion_functions import save_model_outputs
from utility.utilities import FusionData, load_model, create_multiple_features

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_audio_vggish = load_model(AudioVGGishModel, BEST_AUDIO_VGGISH_MODEL, device)
    model_audio_l3 = load_model(AudioL3Model, BEST_AUDIO_L3_MODEL, device)
    model_visual_resnet = load_model(VisualResnetModel, BEST_VISUAL_RESNET_MODEL, device)
    model_visual_l3 = load_model(VisualL3Model, BEST_VISUAL_L3_MODEL, device)

    all_models = [model_audio_vggish, model_audio_l3,
                  model_visual_resnet, model_visual_l3]

    a_vgg_tr, a_vgg_va, a_vgg_te = create_multiple_features(
        feature_files=[AUDIO_VGGISH_TRAIN_FEATURES_FILE,
                       AUDIO_VGGISH_VAL_FEATURES_FILE,
                       AUDIO_VGGISH_TEST_FEATURES_FILE],
        embed_size=AUDIO_VGGISH_EMBED,
        vector_count=VECTORS_PER_AUDIO)

    a_l3_tr, a_l3_va, a_l3_te = create_multiple_features(
        feature_files=[AUDIO_L3_TRAIN_FEATURES_FILE,
                       AUDIO_L3_VAL_FEATURES_FILE,
                       AUDIO_L3_TEST_FEATURES_FILE],
        embed_size=AUDIO_L3_EMBED,
        vector_count=VECTORS_PER_AUDIO)

    v_res_tr, v_res_va, v_res_te = create_multiple_features(
        feature_files=[VISUAL_RESNET_TRAIN_FEATURES_FILE,
                       VISUAL_RESNET_VAL_FEATURES_FILE,
                       VISUAL_RESNET_TEST_FEATURES_FILE],
        embed_size=VISUAL_RESNET_EMBED,
        vector_count=FRAMES_PER_VIDEO)

    v_l3_tr, v_l3_va, v_l3_te = create_multiple_features(
        feature_files=[VISUAL_L3_TRAIN_FEATURES_FILE,
                       VISUAL_L3_VAL_FEATURES_FILE,
                       VISUAL_L3_TEST_FEATURES_FILE],
        embed_size=VISUAL_L3_EMBED,
        vector_count=FRAMES_PER_VIDEO)

    # Models need to be in the same order here, below, and network_tests.py
    train_features = [a_vgg_tr, a_l3_tr, v_res_tr, v_l3_tr]
    train_data_obj = FusionData(train_features)

    save_model_outputs(models=all_models,
                       all_data=train_data_obj.get_data(),
                       device=device,
                       file_out_path=MODELS_TRAIN_OUTPUTS_FILE)

    val_features = [a_vgg_va, a_l3_va, v_res_va, v_l3_va]
    val_data_obj = FusionData(val_features)

    save_model_outputs(models=all_models,
                       all_data=val_data_obj.get_data(),
                       device=device,
                       file_out_path=MODELS_VAL_OUTPUTS_FILE)

    test_features = [a_vgg_te, a_l3_te, v_res_te, v_l3_te]
    test_data_obj = FusionData(test_features)

    save_model_outputs(models=all_models,
                       all_data=test_data_obj.get_data(),
                       device=device,
                       file_out_path=MODELS_TEST_OUTPUTS_FILE)
