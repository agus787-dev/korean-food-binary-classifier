---
title: Korean Food Classifier API
emoji: 🍚
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
---

# Korean Food Classifier API

FastAPI service that classifies a food photo as **bibimbap** or **fried rice** using a CNN built with PyTorch.

## Endpoints

- `GET /health`: check that the server is running and the model is loaded
- `POST /predict`: send an image (`multipart/form-data`, field name `file`) and get the prediction

Example response:

```json
{
  "predicted_class": "bibimbap",
  "confidence": 0.87,
  "probabilities": { "bibimbap": 0.87, "fried_rice": 0.13 }
}
```

Interactive docs: `/docs`

This Space is deployed automatically from the GitHub repository.
