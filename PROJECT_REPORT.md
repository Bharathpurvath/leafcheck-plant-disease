# Project report — Automated Plant Disease Detection Using Deep Learning

## Executive summary

LeafCheck is a browser-based plant-leaf image classifier built with TensorFlow/Keras and Streamlit. It uses ImageNet-pretrained MobileNetV2, returns a predicted class and confidence, displays the top alternatives, and generates Grad-CAM attention overlays. A trained 38-class model and evaluation artifacts are included.

## Problem and intended users

Manual crop inspection can be slow and inconsistent. Farmers, agri-tech teams, agriculture students, and local communities may use image-based screening as an initial crop-health aid. The app is not a professional diagnosis and does not prescribe treatment.

## Dataset and split

The project uses the PlantVillage color subset: 54,305 images across 38 crop/disease or healthy categories. The archive source and attribution are documented in `DATASET_ATTRIBUTION.md`. A fixed seed of 42 produced a stratified, image-level 80/20 test split. The training side was capped at 250 images per class to make CPU training practical; the 9,342-image training/validation pool was split by Keras into 7,474 training and 1,868 validation images. The test partition contains 10,858 images and was not used for fitting or checkpoint selection. This split does not group by physical leaf, so related views of the same leaf may appear across partitions. That can make the reported test score optimistic; use a leaf-grouped split and independent field images to measure generalization more reliably.

## Model and training

The model uses MobileNetV2 pretrained on ImageNet with a global-average-pooling, dropout, and 38-way softmax classifier head. Input images are RGB, resized to 224×224 and rescaled to [-1, 1]. Training augmentation applies random horizontal flips, rotation, zoom, and contrast. The selected final checkpoint used the frozen pretrained backbone and five classifier-head epochs. A one-epoch fine-tuning trial did not improve validation accuracy over the frozen-backbone checkpoint, so it was not selected. The saved model is `models/plant_disease.keras`; class order is in `models/class_names.json`.

## Held-out results

Evaluation on all 10,858 test images produced:

| Metric | Result |
| --- | ---: |
| Accuracy | 92.39% |
| Macro precision | 91.20% |
| Macro recall | 91.12% |
| Macro F1 | 90.73% |
| Weighted F1 | 92.33% |

Per-class precision, recall, F1, and support are in `reports/classification_report.txt`. The test confusion matrix is `reports/confusion_matrix.png`. These are results on a random image-level split of the controlled-background PlantVillage data, not field-photo performance. Because physical-leaf grouping was not enforced, treat these metrics as preliminary and potentially optimistic.

## Application and interpretability

Launch the app with `streamlit run app.py`. Users upload a JPG/PNG leaf image and receive the predicted class, confidence, top three classes, and Grad-CAM heatmap. `gradcam.py` generates a standalone example; `reports/gradcam_example.png` shows a held-out Tomato Late Blight image with 91.0% model confidence. The heatmap indicates model attention, not proof of disease cause.

## Limitations and next steps

PlantVillage images typically have clean backgrounds. Results may not transfer to field photos with clutter, mixed symptoms, or different lighting. Confidence is not a calibrated probability. Before deployment, evaluate field images, measure device latency, test usability with target users, and seek agricultural expert review. Confirm any treatment decision with a local specialist.
