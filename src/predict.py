import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN
from utils import IMAGE_SIZE, MEAN, MODELS_DIR, PREDICTIONS_DIR, STD, TEST_DIR, get_device, load_json


def build_prediction_transform():
    """Use the same resize and normalization as validation/test data."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def predict_single_image(image_path: Path):
    device = get_device()
    class_names = load_json(MODELS_DIR / "class_names.json")

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_cnn_model.pth", map_location=device))
    model.eval()

    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    transform = build_prediction_transform()
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    # Shape becomes [1, 3, 128, 128] because we add a batch dimension.

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]
    confidence_value = confidence.item()

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence_value:.4f}")

    output_image = bgr_image.copy()
    label_text = f"{predicted_class} ({confidence_value:.2f})"
    cv2.putText(
        output_image,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PREDICTIONS_DIR / f"prediction_{image_path.stem}.jpg"
    cv2.imwrite(str(save_path), output_image)
    print("Saved prediction image to:", save_path)


def get_default_test_image():
    """Pick the first available image from the test folder for quick demos."""
    image_paths = sorted(TEST_DIR.glob("*/*.*"))
    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in {TEST_DIR}. Run download_data.py and prepare_data.py first."
        )
    return image_paths[2]


def main():
    parser = argparse.ArgumentParser(description="Predict one image using the trained CNN.")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional path to an image file for prediction.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path) if args.image_path else get_default_test_image()
    print("Using image:", image_path)
    predict_single_image(image_path)


if __name__ == "__main__":
    main()