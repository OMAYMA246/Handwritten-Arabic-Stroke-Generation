# regenerate_knn.py
import torch
import numpy as np
from models.cnn_encoder import ArabicLetterEncoder
from utils.preprocess import preprocess_image
import os

device = torch.device('cpu')

# Charger CNN (encoder.pth de votre binôme)
model = ArabicLetterEncoder(embedding_dim=128)
model.load_state_dict(torch.load('models/encoder.pth', map_location=device))
model.eval()
print('✅ CNN chargé')

# Dossier images ADAB — adapter après téléchargement
IMAGES_DIR = 'dataset_imgs/adab_images_clean/'

# Charger trajectoires existantes depuis ancien npz
old_data    = np.load('data/knn_adab_data.npz')
train_trajs = old_data['train_trajs']  # (12022, 128, 2)
print(f'Trajectoires : {train_trajs.shape}')

# Lister images triées par nom
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
])
print(f'Images trouvées : {len(image_files)}')

# Générer embeddings
embeddings  = []
valid_names = []

for i, fname in enumerate(image_files):
    try:
        path = os.path.join(IMAGES_DIR, fname)
        tensor = preprocess_image(path).to(device)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten()
        embeddings.append(emb)
        valid_names.append(fname)
    except Exception as e:
        print(f'Erreur {fname}: {e}')
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{len(image_files)} traités')

embeddings = np.array(embeddings, dtype=np.float32)
print(f'Embeddings générés : {embeddings.shape}')

# Aligner avec trajectoires
min_len     = min(len(embeddings), len(train_trajs))
embeddings  = embeddings[:min_len]
train_trajs = train_trajs[:min_len]

# Sauvegarder
np.savez(
    'data/knn_adab_data.npz',
    train_embs  = embeddings,
    train_trajs = train_trajs,
    emb_mean    = embeddings.mean(axis=0),
    emb_std     = embeddings.std(axis=0)
)

print(f'\n✅ knn_adab_data.npz régénéré')
print(f'   train_embs  : {embeddings.shape}')
print(f'   train_trajs : {train_trajs.shape}')
print(f'   Norme moyenne : {np.linalg.norm(embeddings, axis=1).mean():.4f}')