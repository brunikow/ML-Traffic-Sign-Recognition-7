"""
Evaluation script for the CBM on the GTSRB test data.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  confusion_matrix, classification_report, precision_recall_fscore_support
import json
import os
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

sys.path.append(str(Path(__file__).resolve().parent.parent))  # to allow imports from overarching directory

from Data.TestData import TestDataset
from Data.TestDataLoader import TestDataLoader
from Models.CBMModel import CBMModel 
from Trainer.CBMTrainer import CBMTrainer
from Models.Model import Model_CNN
from Models.Model2 import Model
from Models.SimpleModel1 import SimpleModel1
from Seeding import set_seed

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        threshold: Threshold for concept predictions
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    # Initialize metrics storage
    all_labels = []
    all_pred_labels = []
    all_concepts = []
    all_pred_concepts = []
    all_concept_probs = []
    
    total_correct_labels = 0
    total_samples = 0
    total_correct_concept_vectors = 0
    total_concept_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (images, (concepts, labels)) in enumerate(test_loader):
            images = images.to(device)
            concepts = concepts.to(device)
            labels = labels.to(device)
            
            # Forward pass
            c_logits, c_probs, l_logits, l_probs = model(images)
            
            # Get predictions
            pred_concepts = (c_probs > threshold).float()
            pred_labels = torch.argmax(l_probs, dim=1)
            
            # Calculate concept vector accuracy (exact match)
            vector_matches = (pred_concepts == concepts).all(dim=1)
            total_correct_concept_vectors += vector_matches.sum().item()
            
            # Calculate per-concept accuracy
            total_concept_predictions += (pred_concepts == concepts).sum().item()
            
            # Calculate label accuracy
            correct_labels = (pred_labels == labels).sum().item()
            total_correct_labels += correct_labels
            
            # Store for overall metrics
            batch_size = labels.size(0)
            total_samples += batch_size
            
            all_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_concepts.extend(concepts.cpu().numpy())
            all_pred_concepts.extend(pred_concepts.cpu().numpy())
            all_concept_probs.extend(c_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_pred_labels = np.array(all_pred_labels)
    all_concepts = np.array(all_concepts)
    all_pred_concepts = np.array(all_pred_concepts)
    all_concept_probs = np.array(all_concept_probs)
    
    # Calculate metrics
    label_accuracy = total_correct_labels / total_samples
    concept_vector_accuracy = total_correct_concept_vectors / total_samples
    concept_accuracy = total_concept_predictions / (total_samples * all_concepts.shape[1])
    
    # Calculate per-concept metrics
    concept_precision, concept_recall, concept_f1, _ = precision_recall_fscore_support(
        all_concepts.flatten(), 
        all_pred_concepts.flatten(), 
        average='binary',
        zero_division=0
    )
    
    # Calculate per-class label metrics
    label_precision, label_recall, label_f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_pred_labels, 
        average='macro',
        zero_division=0
    )
    
    # Create confusion matrix for the labels
    conf_matrix = confusion_matrix(all_labels, all_pred_labels)
    
    # Calculate per-concept accuracy for every concept
    per_concept_acc = []
    for i in range(all_concepts.shape[1]):
        concept_acc = (all_concepts[:, i] == all_pred_concepts[:, i]).mean()
        per_concept_acc.append(concept_acc)
    
    metrics = {
        'label_accuracy': label_accuracy,
        'concept_vector_accuracy': concept_vector_accuracy,
        'concept_accuracy': concept_accuracy,
        'concept_precision': concept_precision,
        'concept_recall': concept_recall,
        'concept_f1': concept_f1,
        'label_precision': label_precision,
        'label_recall': label_recall,
        'label_f1': label_f1,
        'total_samples': total_samples,
        'per_concept_accuracy': per_concept_acc,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:len(conf_matrix)], 
                yticklabels=class_names[:len(conf_matrix)],
                cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_concept_accuracy(per_concept_acc, concept_names, save_path):
    """
    Plot per-concept accuracy.
    """
    plt.figure(figsize=(16, 6))
    bars = plt.bar(range(len(per_concept_acc)), per_concept_acc)
    plt.xticks(range(len(per_concept_acc)), concept_names, rotation=90, ha='right')
    plt.xlabel('Concept')
    plt.ylabel('Accuracy')
    plt.title('Per-Concept Prediction Accuracy')
    plt.ylim(0.98, 1.0) # start at 0.98 because of high accuracies
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def eval():
    """
    Works as the main eval function being called from the main file
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    image_path = DATA_DIR / "GTSRB/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    concept_csv = DATA_DIR / "concepts_per_class.csv"
    label_csv =  DATA_DIR / "GTSRB/GTSRB_Final_Test_GT/GT-final_test.csv"

    destination = Path(MODEL_DIR / "cbmmodel/final_model.pth")
    is_own_model = False
    if destination.exists():
        is_own_model = True
    else:
        destination = (MODEL_DIR / "cbmmodel/final_model_ef.pth")
    
    concept_df = pd.read_csv(concept_csv) 
    concept_names = concept_df.columns[2:].tolist() # needed for plotting
    # Loads the Testdata
    loader = TestDataLoader(image_path=image_path,
                             concept_csv=concept_csv,
                             label_csv=label_csv,
                             pixelsx=128, pixelsy=128,
                             batch_size=32,
                             is_own_model=is_own_model)
    test_loader = loader.get_test_loader()
    # create model
    if (is_own_model):
        cnn_model = SimpleModel1(43).to(device)
    else:
        cnn_model = Model_CNN(43).to(device) # label model
    concept_model = Model(43, 43).to(device)
    model = CBMModel(cnn_model, concept_model).to(device)
    state_dict = torch.load(destination, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    metrics = evaluate_model(model, test_loader, device, threshold=0.5)
    
    print("Evaluation Metrics:")
    print("label_accuracy:", metrics['label_accuracy'])
    print("concept_vector_accuracy:", metrics['concept_vector_accuracy'])
    print("concept_accuracy:", metrics['concept_accuracy'])
    print("concept_precision:", metrics['concept_precision'])
    print("concept_recall:", metrics['concept_recall'])
    print("concept_f1:", metrics['concept_f1'])   
    print("label_precision:", metrics['label_precision'])
    print("label_recall:", metrics['label_recall'])
    print("label_f1:", metrics['label_f1'])
    print("Total samples evaluated:", metrics['total_samples'])
    
    conf_matrix = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        conf_matrix,
        class_names=[str(i) for i in range(43)], 
        save_path= BASE_DIR / "src/CLI/test_visualisation/confusion_matrix.png"
    )
    
    plot_per_concept_accuracy(
        metrics['per_concept_accuracy'],
        concept_names=concept_names,  # Pass the concept names
        save_path= BASE_DIR / "src/CLI/test_visualisation/per_concept_accuracy.png"
    )

if __name__ == "__main__":
    # set seed for reproducability
    set_seed(42)
    # set up device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    image_path = DATA_DIR / "GTSRB/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    concept_csv = DATA_DIR / "concepts_per_class.csv"
    label_csv =  DATA_DIR / "GTSRB/GTSRB_Final_Test_GT/GT-final_test.csv"
    model_path = MODEL_DIR / "cbmmodel/final_model.pth"
    is_own_model = True
    concept_df = pd.read_csv(concept_csv) 
    concept_names = concept_df.columns[2:].tolist() # needed for plotting
    # Loads the Testdata
    loader = TestDataLoader(image_path=image_path,
                             concept_csv=concept_csv,
                             label_csv=label_csv,
                             pixelsx=128, pixelsy=128,
                             batch_size=32,
                             is_own_model=is_own_model)
    test_loader = loader.get_test_loader()
    # create model
    if (is_own_model):
        cnn_model = SimpleModel1(43).to(device)
    else:
        cnn_model = Model_CNN(43).to(device) # label model
    concept_model = Model(43, 43).to(device)
    model = CBMModel(cnn_model, concept_model).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    metrics = evaluate_model(model, test_loader, device, threshold=0.5)
    
    print("Evaluation Metrics:")
    print("label_accuracy:", metrics['label_accuracy'])
    print("concept_vector_accuracy:", metrics['concept_vector_accuracy'])
    print("concept_accuracy:", metrics['concept_accuracy'])
    print("concept_precision:", metrics['concept_precision'])
    print("concept_recall:", metrics['concept_recall'])
    print("concept_f1:", metrics['concept_f1'])   
    print("label_precision:", metrics['label_precision'])
    print("label_recall:", metrics['label_recall'])
    print("label_f1:", metrics['label_f1'])
    print("Total samples evaluated:", metrics['total_samples'])
    
    conf_matrix = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        conf_matrix,
        class_names=[str(i) for i in range(43)], 
        save_path= BASE_DIR / "src/CLI/test_visualisation/confusion_matrix.png"
    )
    
    plot_per_concept_accuracy(
        metrics['per_concept_accuracy'],
        concept_names=concept_names,  # Pass the concept names
        save_path= BASE_DIR / "src/CLI/test_visualisation/per_concept_accuracy.png"
    )