"""
Contains global constants and paths.
Should be placed in the root of the program.
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory of the TAU urban dataset
TAU_DATA_DIR = os.path.join(ROOT_DIR, 'data',
                            'TAU-urban-audio-visual-scenes-2021-development')

# Local data directory
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'local')

# Audio directories
TAU_AUDIO_DIR = os.path.join(TAU_DATA_DIR, 'audio')

# Video directories
TAU_VIDEO_DIR = os.path.join(TAU_DATA_DIR, 'video')

# Image directories
IMAGE_DIR = os.path.join(DATA_DIR, 'image')
IMAGE_TRAIN_DIR = os.path.join(IMAGE_DIR, 'training')
IMAGE_VAL_DIR = os.path.join(IMAGE_DIR, 'validation')
IMAGE_TEST_DIR = os.path.join(IMAGE_DIR, 'testing')
IMAGE_TEMP_DIR = os.path.join(IMAGE_DIR, 'temp')

# CSVs
TAU_CSV_DIR = os.path.join(TAU_DATA_DIR, 'evaluation_setup')
CSV_DIR = os.path.join(DATA_DIR, 'csvs')
CSV_SPLIT_DIR = os.path.join(CSV_DIR, 'splits')
TRAIN_CSV = os.path.join(CSV_DIR, 'train.csv')
TRAIN_SPLIT_CSV = os.path.join(CSV_DIR, 'train_split.csv')
VAL_CSV = os.path.join(CSV_DIR, 'val.csv')
TEST_CSV = os.path.join(CSV_DIR, 'test.csv')
IMAGE_TRAIN_SPLIT_CSV = os.path.join(CSV_DIR, 'image_train_split.csv')
IMAGE_VAL_CSV = os.path.join(CSV_DIR, 'image_val.csv')
IMAGE_TEST_CSV = os.path.join(CSV_DIR, 'image_test.csv')

# Feature directories containing pre-trained model outputs
FEATURE_DATA_DIR = os.path.join(DATA_DIR, 'feature_data')
AUDIO_VGGISH_DATA_DIR = os.path.join(FEATURE_DATA_DIR, 'audio_vggish_features')
AUDIO_L3_DATA_DIR = os.path.join(FEATURE_DATA_DIR, 'audio_l3_features')
VISUAL_RESNET_DATA_DIR = os.path.join(FEATURE_DATA_DIR, 'visual_resnet_features')
VISUAL_L3_DATA_DIR = os.path.join(FEATURE_DATA_DIR, 'visual_l3_features')

# Audio feature files containing pre-trained model outputs
AUDIO_VGGISH_TRAIN_FEATURES_FILE = os.path.join(AUDIO_VGGISH_DATA_DIR, 'audio_vggish_train_data.npz')
AUDIO_VGGISH_VAL_FEATURES_FILE = os.path.join(AUDIO_VGGISH_DATA_DIR, 'audio_vggish_val_data.npz')
AUDIO_VGGISH_TEST_FEATURES_FILE = os.path.join(AUDIO_VGGISH_DATA_DIR, 'audio_vggish_test_data.npz')
AUDIO_L3_TRAIN_FEATURES_FILE = os.path.join(AUDIO_L3_DATA_DIR, 'audio_l3_train_data.npz')
AUDIO_L3_VAL_FEATURES_FILE = os.path.join(AUDIO_L3_DATA_DIR, 'audio_l3_val_data.npz')
AUDIO_L3_TEST_FEATURES_FILE = os.path.join(AUDIO_L3_DATA_DIR, 'audio_l3_test_data.npz')

# Image feature files containing pre-trained model outputs
VISUAL_RESNET_TRAIN_FEATURES_FILE = os.path.join(VISUAL_RESNET_DATA_DIR, 'visual_resnet_train_data.npz')
VISUAL_RESNET_VAL_FEATURES_FILE = os.path.join(VISUAL_RESNET_DATA_DIR, 'visual_resnet_val_data.npz')
VISUAL_RESNET_TEST_FEATURES_FILE = os.path.join(VISUAL_RESNET_DATA_DIR, 'visual_resnet_test_data.npz')
VISUAL_L3_TRAIN_FEATURES_FILE = os.path.join(VISUAL_L3_DATA_DIR, 'visual_l3_train_data.npz')
VISUAL_L3_VAL_FEATURES_FILE = os.path.join(VISUAL_L3_DATA_DIR, 'visual_l3_val_data.npz')
VISUAL_L3_TEST_FEATURES_FILE = os.path.join(VISUAL_L3_DATA_DIR, 'visual_l3_test_data.npz')

# Outputs of the base models (logits)
MODELS_OUTPUT_DIR = os.path.join(FEATURE_DATA_DIR, 'outputs')
MODELS_TRAIN_OUTPUTS_FILE = os.path.join(MODELS_OUTPUT_DIR, 'models_train_outputs.npz')
MODELS_VAL_OUTPUTS_FILE = os.path.join(MODELS_OUTPUT_DIR, 'models_val_outputs.npz')
MODELS_TEST_OUTPUTS_FILE = os.path.join(MODELS_OUTPUT_DIR, 'models_test_outputs.npz')

# Best performing model states
MODEL_DIR = os.path.join(DATA_DIR, 'best_models')
BEST_AUDIO_VGGISH_MODEL = os.path.join(MODEL_DIR, 'best_audio_vggish_model.pkt')
BEST_AUDIO_L3_MODEL = os.path.join(MODEL_DIR, 'best_audio_l3_model.pkt')
BEST_VISUAL_RESNET_MODEL = os.path.join(MODEL_DIR, 'best_visual_resnet_model.pkt')
BEST_VISUAL_L3_MODEL = os.path.join(MODEL_DIR, 'best_visual_l3_model.pkt')
BEST_COMBINER_MODEL = os.path.join(MODEL_DIR, 'best_combiner_model.pkt')

# VGGish variables
VGGISH_PATH = os.path.join(ROOT_DIR, 'models', 'audio', 'vggish')
# VGGISH_PCA_PARAMS = os.path.join(VGGISH_PATH, 'vggish_pca_params.npz')
VGGISH_MODEL = os.path.join(VGGISH_PATH, 'vggish_model.ckpt')

LABELS = {'airport': 0,
          'bus': 1,
          'metro': 2,
          'metro_station': 3,
          'park': 4,
          'public_square': 5,
          'shopping_mall': 6,
          'street_pedestrian': 7,
          'street_traffic': 8,
          'tram': 9
          }

LABEL_COUNT = len(LABELS)
SEED = 42                   # Seed for randomizers
EPOCHS_AFTER_NEW_BEST = 8   # Stop training a network early if it does not improve
AUDIO_L3_EMBED = 512        # OpenL3 supports 512 and 6144
AUDIO_VGGISH_EMBED = 128    # Determined by VGGish, not this variable
VISUAL_RESNET_EMBED = 2048  # Penultimate layer, not determined by this variable
VISUAL_L3_EMBED = 512       # OpenL3 supports 512 and 8192

# The following variables should be separated for each base classifier,
# if they have a different amount of embeddings for a single file
FRAMES_PER_VIDEO = 2        # Amount of extracted frames per video
VECTORS_PER_AUDIO = 10      # Determined by VGGish and OpenL3, not this variable
