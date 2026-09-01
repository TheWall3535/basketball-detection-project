"""
detection_client.py -- client for the basketball detector service.

This is the CLIENT (how the deployed service is used), not a unit test. It POSTs
an image and prints JSON detections; with --save, also fetches the annotated
image and writes it to disk. Torch-free -- it just talks to the running server
over HTTP.

Usage:
    python detection_client.py shot.jpg
    python detection_client.py shot.jpg --save out.png
    python detection_client.py shot.jpg --url http://localhost:5000/predict
"""

import argparse
import requests


def main():
    parser = argparse.ArgumentParser(description="POST an image to the detector")
    parser.add_argument('image', help="image file path")
    parser.add_argument('--url', default='http://localhost:5000/predict')
    parser.add_argument('--save', default=None,
                        help="also fetch the annotated image and save here")
    args = parser.parse_args()

    # JSON detections
    try:
        with open(args.image, 'rb') as f:
            resp = requests.post(args.url, files={'file': (args.image, f)}, timeout=60)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: could not connect. Is app.py running at {args.url}?")
        return

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    print(f"checkpoint: {data.get('checkpoint')}  |  {data.get('count')} detection(s)")
    for d in data.get('detections', []):
        x1, y1, x2, y2 = d['box']
        print(f"  {d['label']:<18} {d['score']:.3f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    # optionally fetch + save the annotated image
    if args.save:
        with open(args.image, 'rb') as f:
            r2 = requests.post(args.url + "?annotated=true",
                               files={'file': (args.image, f)}, timeout=60)
        if r2.status_code == 200:
            with open(args.save, 'wb') as out:
                out.write(r2.content)
            print(f"annotated image saved -> {args.save}")
        else:
            print(f"annotated fetch failed: HTTP {r2.status_code}")


if __name__ == '__main__':
    main()
