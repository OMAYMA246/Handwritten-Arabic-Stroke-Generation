from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
import time
import os

app = Flask(__name__)

# KNN Calliar
from services.knn_calliar_service import KNNCalliarService
knn_service = KNNCalliarService(npz_path='data/calliar_knn_data.npz')

# CNN
cnn_service = None
if os.path.exists('models/encoder.pth'):
    try:
        from services.cnn_service import CNNService
        cnn_service = CNNService()
        print('✅ CNN actif')
    except Exception as e:
        print(f'⚠️  CNN erreur : {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        t_start = time.time()
        body = request.get_json()
        if not body or 'image' not in body:
            return jsonify({'error': 'Aucune image'}), 400

        image_data = body['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # CNN → embedding
        if cnn_service is not None:
            embedding = cnn_service.get_embedding(image_bytes)
        else:
            embedding = np.random.randn(128).astype(np.float32)

        # KNN → trajectoire (128, 2)
        trajectory, distance, confidence = knn_service.find_trajectory(embedding)

        # Convertir en strokes séparés selon les sauts
        def traj_to_strokes(traj, threshold=0.05):
            """
            Diviser la trajectoire en strokes selon les sauts
            threshold : distance normalisée entre 0 et 1
            """
            strokes  = []
            current  = [[float(traj[0][0]), float(traj[0][1])]]

            for i in range(1, len(traj)):
                dx   = traj[i][0] - traj[i-1][0]
                dy   = traj[i][1] - traj[i-1][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist > threshold:
                    if len(current) > 1:
                        strokes.append(current)
                    current = [[float(traj[i][0]), float(traj[i][1])]]
                else:
                    current.append([float(traj[i][0]), float(traj[i][1])])

            if len(current) > 1:
                strokes.append(current)

            return strokes if strokes else [[[float(p[0]), float(p[1])]
                                             for p in traj]]

        strokes = traj_to_strokes(trajectory)

        return jsonify({
            'points'    : trajectory.tolist(),  # (128, 2) pour export JSON
            'strokes'   : strokes,              # liste de traits séparés
            'time'      : round(time.time() - t_start, 3),
            'confidence': confidence,
            'distance'  : distance,
            'mode'      : 'cnn' if cnn_service else 'test'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

embedding = cnn_service.get_embedding(image_bytes)

# DEBUG — ajoute ces 3 lignes
print(f"Embedding std  : {embedding.std():.4f}")
print(f"Embedding mean : {embedding.mean():.4f}")
print(f"Embedding norm : {np.linalg.norm(embedding):.4f}")