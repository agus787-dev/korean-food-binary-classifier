"""Load the trained CNN and predict the class of one image.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from model import SimpleCNN
from preprocessing_data import get_eval_transforms
from utils import MODELS_DIR, get_device

MODEL_PATH = MODELS_DIR / "best_cnn_model.pth"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


class Predictor:
    """Loads the model once and can then predict many images quickly."""

    def __init__(self, model_path: Path = MODEL_PATH, class_names_path: Path = CLASS_NAMES_PATH):
        self.device = get_device()

        # class_names.json was saved by train.py, e.g. ["bibimbap", "fried_rice"]
        with open(class_names_path, encoding="utf-8") as f:
            self.class_names = json.load(f)

        self.model = SimpleCNN(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # inference mode

        # Must be the same transforms as validation/test during training
        self.transform = get_eval_transforms()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Predict one PIL image. Returns class name, confidence and all probabilities."""
        # Force 3 channels: PNG with alpha or grayscale images would break the model input
        image = image.convert("RGB")

        # [3, H, W] -> [1, 3, H, W]: the model expects a batch dimension
        x = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]  # [num_classes], sums to 1

        best_idx = int(torch.argmax(probs))
        return {
            "predicted_class": self.class_names[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": {
                name: float(p) for name, p in zip(self.class_names, probs)
            },
        }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    predictor = Predictor()
    result = predictor.predict(Image.open(image_path))

    print(f"Image: {image_path}")
    print(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
    for name, p in result["probabilities"].items():
        print(f"  {name:11s} {p:.4f}")


if __name__ == "__main__":
    main()