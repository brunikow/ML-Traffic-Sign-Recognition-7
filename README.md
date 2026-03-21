# Traffic Sign Recognition – Getting Started

This project trains a traffic sign recognition model using the GTSRB dataset.
Created for our Semesterproject at the Humboldt University Berlin.

---

## 1️. Install Conda

Download and install **Anaconda** or **Miniconda**:

[https://www.anaconda.com/download](https://www.anaconda.com/download)

---

## 2️. Set up project environment

From the project root:

```
# Create the conda environment
conda env create -f config/env.yaml

# Activate the environment
conda activate gr7-ml
```

---

## 3. Initialize

The project uses the **GTSRB dataset**. To properly train our Models we need to download the dataset:
```
python src/main.py init
```

After running this command following folders will appear in your file structure:

-   `GTSRB_Final_Training_Images`
-   `GTSRB_Final_Test_Images`
-   `GTSRB_Final_Test_GT`

--- 

## 4. Train the model
```
python src/main.py run

```
This command can be followed by the following configs:
```
python src/main.py run [command]
```
own_model: runs our model instead of the EfficientNetV2

This will:

-   Train the CNN
-   Evaluate performance per batch
-   Compute precision, recall, and F1 scores

---

## 5. Evaluate the model