"""
app.py -- Flask microservice for the basketball detector.

POST an image to /predict; get JSON detections by default, or the annotated
image (boxes drawn) when ?annotated=true is passed. The model/predictor is
built ONCE at startup and reused per request.

Run:
    python app.py --checkpoint basketball_best.pt
    python app.py --checkpoint basketball_run2.pt --score-threshold 0.4
Then:
    curl -F file=@shot.jpg http://localhost:5000/predict
    curl -F file=@shot.jpg "http://localhost:5000/predict?annotated=true" --output out.png
"""

import os
import argparse
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch

from src.detection_serving import (
    DetectionPredictor, draw_detections, image_to_png_bytes)

app = Flask(__name__)

predictor = None
checkpoint_name = None


@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'model not loaded'}), 500

    f = request.files.get('file') or request.files.get('image')
    if f is None:
        return jsonify({'error': "no image uploaded; POST a file under key "
                                 "'file' (multipart/form-data)"}), 400

    try:
        image = Image.open(f.stream).convert('RGB')
        dets = predictor.predict(image)
    except Exception as e:
        return jsonify({'error': f'inference failed: {type(e).__name__}: {e}'}), 500

    # annotated image on request, else JSON
    want_annotated = request.args.get('annotated', 'false').lower() in ('true', '1', 'yes')
    if want_annotated:
        annotated = draw_detections(image, dets)
        png = image_to_png_bytes(annotated)
        import io
        return send_file(io.BytesIO(png), mimetype='image/png')

    return jsonify({
        'checkpoint': checkpoint_name,
        'count': len(dets),
        'detections': dets,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'checkpoint': checkpoint_name,
                    'loaded': predictor is not None})


def main():
    global predictor, checkpoint_name

    parser = argparse.ArgumentParser(description="Basketball detector service")
    parser.add_argument('--checkpoint', required=True,
                        help="path to the trained model state_dict (.pt)")
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cpu',
                        help="'cpu' or 'cuda' (cpu is fine for single-image serving)")
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but no GPU available; falling back to CPU.")
        args.device = 'cpu'

    checkpoint_name = os.path.basename(args.checkpoint)
    predictor = DetectionPredictor(
        args.checkpoint, device=args.device,
        score_threshold=args.score_threshold)

    print("=" * 60)
    print(f"  BASKETBALL DETECTOR SERVING")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  score threshold: {args.score_threshold}")
    print(f"  device: {args.device}")
    print(f"  endpoint: http://{args.host}:{args.port}/predict  (POST an image)")
    print(f"           add ?annotated=true for the drawn image")
    print("=" * 60, flush=True)

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
