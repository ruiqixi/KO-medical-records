import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

# ========== STEP 1: Load the Saved Model ==========
print("Loading saved ClinicalBERT model...")
model_path = "clinicalbert_patient_outcome"  # Path to the saved model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# ========== STEP 2: Load the Validation Dataset ==========
print("Loading validation dataset...")
file_path = "preprocessed_clinical_notes_old.csv"  # Use the old dataset
df = pd.read_csv(file_path)

# Ensure no missing values
df.dropna(subset=["Clinical_Note", "Outcome"], inplace=True)

# Convert outcomes to numerical labels
label_mapping = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
df["label"] = df["Outcome"].map(label_mapping)

# Load validation set (20% of old dataset, as done during training)
from sklearn.model_selection import train_test_split
_, val_texts, _, val_labels = train_test_split(
    df["Clinical_Note"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tokenize validation data
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# ========== STEP 3: Convert to PyTorch Dataset ==========
class ClinicalNotesDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

val_dataset = ClinicalNotesDataset(val_encodings, val_labels)

# Create DataLoader for validation set
batch_size = 8
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ========== STEP 4: Evaluate the Model on Validation Set ==========
print("Evaluating on validation set...")

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

# Compute accuracy
val_accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy (Old Dataset 2019-2021): {val_accuracy:.4f}")
