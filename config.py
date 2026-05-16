import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'encoder.pth')
KNN_DATA_PATH  = os.path.join(DATA_DIR,   'knn_adab_data.npz')

# Paramètres CNN — identiques au notebook de votre binôme
IMAGE_SIZE    = 224
EMBEDDING_DIM = 128

# Paramètres KNN
KNN_NEIGHBORS = 3