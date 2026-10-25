# LeafCheck — Plant disease image screening

A runnable deep-learning project for classifying plant leaf images. It includes a trained 38-class model, a Streamlit app, training and evaluation scripts, class-wise metrics, a confusion matrix, and a Grad-CAM example.

## Dataset

This project uses PlantVillage's 54,305 color images across 38 class folders. The public archive source and CC BY-SA 3.0 attribution are in [DATASET_ATTRIBUTION.md](DATASET_ATTRIBUTION.md). The 2.18 GB archive is not included in the source ZIP. It has been downloaded and extracted during this project run; download it again from the linked source if you want to retrain elsewhere.

## Run the included app

Python 3.10–3.12 is recommended. In a virtual environment:

```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app loads `models/plant_disease.keras` and `models/class_names.json` from the project directory.

## Recreate training and evaluation

After extracting the dataset so `raw/color/` contains the class folders:

```
python prepare_data.py --data path/to/dataset/raw/color --output data/split --max-trainval-per-class 250
python train.py --data data/split/trainval --epochs 5 --fine-tune-epochs 0
python evaluate.py --data data/split/test
python gradcam.py --image path/to/leaf.jpg
```

`prepare_data.py` uses seed 42 to make a stratified, image-level 80/20 test split, then caps the training/validation pool per class to keep CPU training practical. The trainer uses 20% of that pool for validation. The included selected checkpoint was trained with five classifier-head epochs and a frozen ImageNet MobileNetV2 backbone; a one-epoch fine-tuning pass did not beat its validation score. The test partition contains 10,858 images and was not used for fitting or checkpoint selection. This is an image-level split: related views of the same physical leaf may cross partitions, so the score may be optimistic. A leaf-grouped split and external field-photo evaluation are needed for stronger generalization claims.

Evaluation writes `reports/classification_report.txt` and `reports/confusion_matrix.png`. The random image-level test scores are 92.39% accuracy, 91.20% macro precision, 91.12% macro recall, and 90.73% macro F1. See `PROJECT_REPORT.md` for the full method and limits.

## Model and app behavior

Images are resized to 224×224. Training augmentation includes random horizontal flip, rotation, zoom, and contrast. The app displays class confidence, the top three predictions, and a Grad-CAM heatmap. `reports/gradcam_example.png` is a generated held-out example. A prediction is an initial screening aid; it is not a professional diagnosis, its confidence is not calibrated, and the app does not prescribe pesticides.

## Limitations

PlantVillage photographs typically have clean backgrounds. Results may not transfer to field photos with clutter, mixed symptoms, or different lighting. The image-level split may also place views of the same physical leaf in both training and test groups, making metrics optimistic. Validate on leaf-grouped and independent field images, and consult an agricultural specialist before making treatment decisions.

## Deploy on Streamlit Community Cloud

The project is ready for Streamlit Community Cloud. Put the contents of this folder in a public GitHub repository, then create an app at [share.streamlit.io](https://share.streamlit.io) using `app.py` as the entry point. Choose Python 3.12 in Advanced settings. The trained model is included; the full 2.18 GB training archive is not.
