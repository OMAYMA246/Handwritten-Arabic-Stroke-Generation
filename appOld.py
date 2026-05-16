from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
import time
import os
import sys

app = Flask(__name__)

# ── Initialiser KNN (toujours disponible)
from services.knn_service import KNNService
knn_service = KNNService()

# ── Initialiser CNN (seulement si encoder.pth existe)
cnn_service = None
if os.path.exists('models/encoder.pth'):
    try:
        from services.cnn_service import CNNService
        cnn_service = CNNService()
        print('✅ Mode production — CNN actif')
    except Exception as e:
        print(f'⚠️  Erreur CNN : {e}')
else:
    print('⚠️  encoder.pth non trouvé — mode test (embedding aléatoire)')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'cnn'  : cnn_service is not None,
        'knn'  : True,
        'mode' : 'production' if cnn_service else 'test'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        t_start = time.time()

        # ── Récupérer l'image (base64 depuis index.html)
        body = request.get_json()
        if not body or 'image' not in body:
            return jsonify({'error': 'Aucune image reçue'}), 400

        # Décoder base64
        # Format : "data:image/png;base64,xxxxxx"
        image_data = body['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # ── CNN → embedding
        if cnn_service is not None:
            embedding = cnn_service.get_embedding(image_bytes)
        else:
            # Mode test — embedding aléatoire
            print('⚠️  Embedding aléatoire (mode test)')
            embedding = np.random.randn(128).astype(np.float32)

        # ── KNN → trajectoire
        trajectory, distance, confidence = knn_service.find_trajectory(embedding)

        t_end = time.time()

        # Retourner au format attendu par index.html
        # index.html attend : data.points, data.time, data.confidence
        return jsonify({
            'points'    : trajectory.tolist(),  # [[x,y], [x,y], ...]
            'time'      : round(t_end - t_start, 3),
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