import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ========== STEP 1: Data Loading and Preparation ==========
print("Loading dataset...")

# Load dataset
file_path = "preprocessed_clinical_notes_old.csv"
df = pd.read_csv(file_path)

# Check class distribution
print("\nClass distribution:")
print(df["Outcome"].value_counts())

# Convert outcomes to numerical labels
label_mapping = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
df["label"] = df["Outcome"].map(label_mapping)

# Split data ensuring class balance
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Clinical_Note"], df["label"], 
    test_size=0.15, random_state=42, stratify=df["label"]
)

print(f"Training set: {len(train_texts)} samples")
print(f"Validation set: {len(val_texts)} samples")

# ========== STEP 2: Data Tokenization ==========
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Check the maximum length needed - use all available samples instead of sampling
sample_lengths = [len(tokenizer.encode(text)) for text in df["Clinical_Note"]]
print(f"\nText length stats: Min={min(sample_lengths)}, Mean={np.mean(sample_lengths):.1f}, Max={max(sample_lengths)}")

# Set max_length based on distribution
max_length = min(192, int(np.percentile(sample_lengths, 95)))
print(f"Using max_length={max_length} (95th percentile)")

# Tokenize texts
train_encodings = tokenizer(
    train_texts.tolist(), 
    truncation=True, 
    padding="max_length", 
    max_length=max_length,
    return_tensors="pt"
)

val_encodings = tokenizer(
    val_texts.tolist(), 
    truncation=True, 
    padding="max_length", 
    max_length=max_length,
    return_tensors="pt"
)

# ========== STEP 3: Dataset Creation ==========
class ClinicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = ClinicalDataset(train_encodings, train_labels.tolist())
val_dataset = ClinicalDataset(val_encodings, val_labels.tolist())

# ========== STEP 4: Model Setup ==========
print("\nSetting up model...")

# Use GPU if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Load model
model = BertForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", 
    num_labels=3,
    hidden_dropout_prob=0.2,  # Increased dropout for better generalization
    attention_probs_dropout_prob=0.2
)
model.to(device)

# ========== STEP 5: Training Configuration ==========
batch_size = 16
epochs = 8
learning_rate = 2e-5
weight_decay = 0.01  # L2 regularization

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ========== STEP 6: Training Loop ==========
print("\nStarting training...")

# Initialize tracking variables
train_losses = []
val_losses = []
val_accuracies = []
best_accuracy = 0
best_f1 = 0
best_state = None

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update tracking
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Track loss
            val_loss += outputs.loss.item()
            
            # Track predictions
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    accuracy = accuracy_score(true_labels, predictions)
    val_accuracies.append(accuracy)
    
    f1 = f1_score(true_labels, predictions, average='macro')
    
    # Print results
    print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Class report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                               target_names=list(label_mapping.keys()), digits=4))
    
    # Check if current model is the best
    if f1 > best_f1:
        best_f1 = f1
        best_accuracy = accuracy
        best_state = model.state_dict().copy()
        print(f"New best model saved with F1: {best_f1:.4f}")

# Load best model
if best_state:
    model.load_state_dict(best_state)
    print(f"Loaded best model with Accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f}")

# ========== STEP 7: Plot Training Metrics ==========
try:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Saved training metrics plot to 'training_metrics.png'")
except Exception as e:
    print(f"Error creating plots: {e}")

# ========== STEP 8: Evaluate on New Dataset ==========
print("\nEvaluating on new dataset...")

# Load new dataset
new_file_path = "preprocessed_clinical_notes_new.csv"
df_new = pd.read_csv(new_file_path)
df_new["label"] = df_new["Outcome"].map(label_mapping)

print(f"New dataset size: {len(df_new)} samples")
print("Class distribution:")
print(df_new["Outcome"].value_counts())

# Tokenize
new_encodings = tokenizer(
    df_new["Clinical_Note"].tolist(), 
    truncation=True, 
    padding="max_length", 
    max_length=max_length,
    return_tensors="pt"
)

# Create dataset and loader
new_dataset = ClinicalDataset(new_encodings, df_new["label"].tolist())
new_loader = DataLoader(new_dataset, batch_size=batch_size)

# Evaluate
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(new_loader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

# Calculate performance
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='macro')

print(f"New Dataset - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

# Try to visualize confusion matrix
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.keys()),
                yticklabels=list(label_mapping.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on New Dataset')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to 'confusion_matrix.png'")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, 
                           target_names=list(label_mapping.keys()), digits=4))

# ========== STEP 9: Save Model for Snorkel ==========
print("\nSaving model for Snorkel pipeline...")
model.save_pretrained("clinicalbert_for_snorkel")
tokenizer.save_pretrained("clinicalbert_for_snorkel")

# Save metadata about the model
with open("clinicalbert_for_snorkel/model_info.txt", "w") as f:
    f.write(f"Best Validation Accuracy: {best_accuracy:.4f}\n")
    f.write(f"Best Validation F1 Score: {best_f1:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test F1 Score: {f1:.4f}\n")
    f.write("\nTraining Parameters:\n")
    f.write(f"- Batch Size: {batch_size}\n")
    f.write(f"- Learning Rate: {learning_rate}\n")
    f.write(f"- Max Sequence Length: {max_length}\n")
    f.write(f"- Weight Decay: {weight_decay}\n")
    f.write(f"- Training Epochs: {epochs}\n")

print("Model and metadata saved. Ready for Snorkel pipeline implementation.")