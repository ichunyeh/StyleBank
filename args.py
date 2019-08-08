import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

SEED = 5

batch_size = 4
lr = 0.001
T = 2
CONENT_WEIGHT = 1
STYLE_WEIGHT = 1000000
REG_WEIGHT = 1e-5

IMG_SIZE = 512
CONTENT_IMG_DIR = 'data/coco'
STYLE_IMG_DIR = 'data/style'
NEW_STYLE_IMG_DIR = 'data/new_style'
TEST_IMG_DIR = 'data/test'

K = 1000
MAX_ITERATION = 300 * K
ADJUST_LR_ITER = 10 * K
LOG_ITER = 1 * K

alpha = 1

param_name = '_nnResize_128_6'  # +'_alpha05'
OUTPUT_IMG_DIR = 'output/' + param_name
LOG_DIR = 'log/' + param_name
MODEL_WEIGHT_DIR = 'weights/' + param_name

BANK_WEIGHT_DIR = os.path.join(MODEL_WEIGHT_DIR, 'bank')
BANK_WEIGHT_PATH = os.path.join(BANK_WEIGHT_DIR, '{}.path')
NEW_BANK_WEIGHT_DIR = os.path.join(MODEL_WEIGHT_DIR, 'bank')
NEW_BANK_WEIGHT_PATH = os.path.join(BANK_WEIGHT_DIR, '{}.path')
MODEL_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'model.pth')
ENCODER_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'encoder.pth')
DECODER_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'decoder.pth')
GLOBAL_STEP_PATH = os.path.join(MODEL_WEIGHT_DIR, 'global_step.log')
