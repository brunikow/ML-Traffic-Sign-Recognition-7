# Getting Started

This project trains a traffic sign recognition model using the GTSRB dataset.
Created for our semester project at the Humboldt University Berlin.

## 1. Clone the Repo

```
git clone https://github.com/brunikow/ML-Traffic-Sign-Recognition-7.git tsr-gr7
```

---

## 2. Install Conda

Download and install **Anaconda** or **Miniconda**:

[https://www.anaconda.com/download](https://www.anaconda.com/download)
---

## 3. Set up project environment

From the project root:

```
# Create the conda environment
conda env create -f config/env.yaml

# Activate the environment
conda activate gr7-ml
```

---

## 4. Initialize

The project uses the **GTSRB dataset**. To properly train our Models we need to download the dataset:
```
python src/main.py init
```

After running this command following folders will appear in your file structure:

-   `GTSRB_Final_Training_Images`
-   `GTSRB_Final_Test_Images`
-   `GTSRB_Final_Test_GT`

--- 

## 5. Train the model
```
python src/main.py run
```
This command can be followed by the following configs:
```
python src/main.py run [command]
```
own_model: runs our model instead of the EfficientNetV2

This will:

- Train the CNN
- Evaluate performance per batch
- Compute precision, recall, and F1 scores

---

## 6. Evaluate the model
```
python src/main.py eval
```

This will:

- Load the test dataset and trained model.
- Evaluate the model and compute:
- Label accuracy
- Concept vector accuracy
- Concept accuracy, precision, recall, F1

Save visualizations:
- Confusion matrix (confusion_matrix.png)
- Per-concept accuracy (per_concept_accuracy.png)

Metrics are printed to the console and saved plots are in `src/CLI/test_visualisation/`.