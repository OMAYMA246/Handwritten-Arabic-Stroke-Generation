import torch
import numpy as np
from models.cnn_encoder import ArabicLetterEncoder
from utils.preprocess import preprocess_image
from config import CNN_MODEL_PATH, EMBEDDING_DIM

class CNNService:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Créer le modèle avec l'architecture de votre binôme
        self.model = ArabicLetterEncoder(embedding_dim=EMBEDDING_DIM)

        # Charger ses poids
        state_dict = torch.load(CNN_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        print(f'✅ CNN chargé depuis {CNN_MODEL_PATH}')
        print(f'   Device : {self.device}')

    def get_embedding(self, image_input):
        """
        image_input : fichier Flask ou bytes
        retourne    : numpy array (128,)
        """
        tensor = preprocess_image(image_input).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.cpu().numpy().flatten()  # (128,)