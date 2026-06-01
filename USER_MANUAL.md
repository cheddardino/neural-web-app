# DressCheck User Manual

## Overview

DressCheck is a web-based government employee attire compliance checker for upper-garment images. A user uploads an image, the system analyzes it using a trained HOG + SVM model, and the website returns whether the image is classified as `Compliant` or `Non-Compliant`.

This manual is intended for project presentation, demonstration, and basic user guidance.

## System Requirements

For website users:

- A modern web browser such as Chrome, Edge, Safari, or Firefox
- An image file of a government employee upper garment or outfit
- Supported image formats: JPG, PNG, WEBP, and other browser-supported image files

For local developers or presenters:

- Python 3.9 or newer
- Project dependencies from `requirements.txt`
- Model files in the `models/` folder:
  - `svm_model.pkl`
  - `scaler.pkl`

## Project Files Used by the App

- `index.html`: Main website interface
- `api/predict.py`: Prediction API endpoint
- `models/svm_model.pkl`: Trained SVM classifier
- `models/scaler.pkl`: Feature scaler used before prediction
- `Midterm Dataset/`: Training dataset
- `DressCheck_Training_Notebook.ipynb`: Training and evaluation notebook

## How to Use the Website

1. Open the DressCheck website.
2. Click the upload area labeled `Upload garment image`, or drag and drop an image into the upload area.
3. Confirm that the image preview appears on the page.
4. Click `Analyze Attire`.
5. Wait for the scan to finish.
6. Review the result card.

## Understanding the Result

The result card shows:

- `Compliant`: The uploaded image is classified as acceptable based on the trained model.
- `Non-Compliant`: The uploaded image is classified as not acceptable based on the trained model.
- `Confidence`: The model's confidence score for the selected prediction.
- `Verdict`: A simplified pass/fail interpretation:
  - `Allowed`: The image is classified as compliant.
  - `Violation`: The image is classified as non-compliant.

## How to Check Another Image

After a prediction appears:

1. Click `Check another image`.
2. Upload or drag another image.
3. Click `Analyze Attire` again.

## Recommended Image Input

For best results:

- Use a clear image of the upper garment.
- Avoid heavily blurred or very dark images.
- Make sure the clothing is visible and not blocked by objects.
- Use images similar in style to the training dataset.

## Local Demo Instructions

Use these steps when presenting the project locally.

1. Open a terminal in the project folder.
2. Install dependencies if needed:

```bash
pip install -r requirements.txt
```

3. For a full website demo, use the deployed Vercel site or run the project through Vercel's local development server if the Vercel CLI is installed:

```bash
vercel dev
```

4. For API-only testing, run the Flask endpoint:

```bash
flask --app api.predict run
```

The frontend sends uploaded images to `/api/predict`, so the complete website demo should be served from the same app environment as the API.

## Training Notebook Instructions

The notebook file is:

```text
DressCheck_Training_Notebook.ipynb
```

Use it to show:

- Dataset class counts
- Image preprocessing
- HOG feature extraction
- SVM training with grid search
- Accuracy and classification report
- Exporting `svm_model.pkl` and `scaler.pkl`

Before running the notebook, make sure the dataset folders exist:

```text
Midterm Dataset/Compliant
Midterm Dataset/Non-Compliant
```

## Troubleshooting

### No image preview appears

Make sure the selected file is an image file. Try using JPG or PNG.

### Prediction fails

Possible causes:

- The API is not running or not deployed correctly.
- `models/svm_model.pkl` is missing.
- `models/scaler.pkl` is missing.
- The uploaded file is corrupted or unsupported by the server.

### API returns an error about missing model files

Check that both model files are inside the `models/` folder.

### Result seems incorrect

The system depends on the training dataset and learned image patterns. Try using a clearer image or an image closer to the dataset examples.

## Presentation Checklist

- Show the dataset folders and class counts.
- Open `DressCheck_Training_Notebook.ipynb` and explain the training workflow.
- Show the website interface for government employee attire checking.
- Upload a sample compliant image.
- Upload a sample non-compliant image.
- Show this user manual as project documentation.

## Limitations

- The model is trained on a limited image dataset.
- It may not generalize perfectly to all clothing styles, camera angles, or lighting conditions.
- It checks visual similarity based on image features, not official human judgment.
- It should be used as a project prototype, not as a final authority for government employee dress code enforcement.
