# Korean Food Binary Classifier

A PyTorch image classification project that distinguishes between two visually similar Korean dishes:

- `bibimbap`
- `fried_rice`

This is a small binary classification task designed to practice CNN training, image preprocessing, evaluation, and deployment as a simple web API.

## Project overview

The project trains a lightweight convolutional neural network from image data and exposes predictions through a FastAPI service. It includes:

- dataset preparation and image transforms
- model training and validation
- evaluation and plotting
- a prediction script for single-image inference
- a FastAPI app for serving the trained model

## Repository structure

```text
.
├── app/
│   └── main.py              # FastAPI app for inference
├── data/                    # raw or processed datasets
├── dataset/                 # dataset-related assets or labels
├── deploy/                  # deployment-related files
├── notebooks/               # experimentation notebooks
├── outputs/
│   ├── models/              # trained model artifacts
│   └── plots/               # training curves and plots
├── src/
│   ├── download_data.py     # helper for downloading data
│   ├── evaluate.py          # evaluation utilities
│   ├── model.py             # CNN architecture
│   ├── predict.py           # single-image prediction entry point
│   ├── preprocessing_data.py# image transform setup
│   ├── train.py             # training pipeline
│   └── utils.py             # utility functions
├── Dockerfile               # container setup for deployment
├── requirements.txt         # runtime dependencies
├── requirements_full.txt    # full dependency list
└── ReadMe.md                # project documentation
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv myenv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using a GPU-capable environment, install the appropriate PyTorch build for your system instead of the default CPU version.

## Training the model

From the project root, run:

```bash
python src/train.py --epochs 5 --batch_size 16 --learning_rate 0.001
```

This will:

- build the dataset loaders
- train the CNN
- save the best model to `outputs/models/best_cnn_model.pth`
- save class names to `outputs/models/class_names.json`
- generate training plots in `outputs/plots/`

## Running inference

Predict a single image:

```bash
python src/predict.py path/to/image.jpg
```

Example output:

```text
Image: path/to/image.jpg
Prediction: bibimbap (92.3%)
  bibimbap    0.9230
  fried_rice  0.0770
```

## Running the API locally

Start the FastAPI app:

```bash
uvicorn app.main:app --reload --port 8000
```

Then open:

- http://localhost:8000/docs
- http://localhost:8000/health

You can upload an image to the `/predict` endpoint to receive a prediction result.

## Docker

The project includes a Dockerfile for deployment. Build and run it with:

```bash
docker build -t korean-food-classifier .
docker run -p 7860:7860 korean-food-classifier
```

## Notes

- The project uses a small, fine-grained food recognition task with similar-looking classes.
- Model training and inference are intentionally lightweight and approachable for learning or prototyping.
- The API accepts image files in JPEG, PNG, or WebP format, with a 5 MB maximum size.

## License

This project does not include a license file yet. Add one if you plan to distribute or publish the project.
