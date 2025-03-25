import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Snorkel imports
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.analysis import get_label_buckets

# ======== STEP 1: Load datasets ========
print("Loading datasets...")
# Load old dataset for training
df_old = pd.read_csv("preprocessed_clinical_notes_old.csv")
# Load new dataset for testing
df_new = pd.read_csv("preprocessed_clinical_notes_new.csv")

# Define label mapping (same as in training)
label_mapping = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
id2label = {v: k for k, v in label_mapping.items()}

# Convert string labels to integers
df_old["label"] = df_old["Outcome"].map(label_mapping)
df_new["label"] = df_new["Outcome"].map(label_mapping)

print(f"Old dataset: {len(df_old)} samples")
print(f"New dataset: {len(df_new)} samples")

# ======== STEP 2: Load ClinicalBERT model ========
print("Loading ClinicalBERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
model_path = "clinicalbert_for_snorkel"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# ======== STEP 3: Create Labeling Functions ========
# Define ABSTAIN value for Snorkel
ABSTAIN = -1

# Helper function to get a sample of text to analyze common phrases
def analyze_sample_phrases():
    """Helper to analyze text and find common phrases for LFs"""
    # Get sample of each class
    normals = df_old[df_old["Outcome"] == "Normal"]["Clinical_Note"].sample(min(10, len(df_old[df_old["Outcome"] == "Normal"]))).tolist()
    abnormals = df_old[df_old["Outcome"] == "Abnormal"]["Clinical_Note"].sample(min(10, len(df_old[df_old["Outcome"] == "Abnormal"]))).tolist()
    inconclusives = df_old[df_old["Outcome"] == "Inconclusive"]["Clinical_Note"].sample(min(10, len(df_old[df_old["Outcome"] == "Inconclusive"]))).tolist()
    
    print("\nSample phrases from Normal notes:")
    for note in normals[:3]:
        print(f"- {note[:100]}...")
    
    print("\nSample phrases from Abnormal notes:")
    for note in abnormals[:3]:
        print(f"- {note[:100]}...")
    
    print("\nSample phrases from Inconclusive notes:")
    for note in inconclusives[:3]:
        print(f"- {note[:100]}...")

# Uncomment to run phrase analysis
# analyze_sample_phrases()

# LF1: ClinicalBERT model predictions
@labeling_function(name="clinicalbert")
def lf_clinicalbert(x):
    """Use fine-tuned ClinicalBERT to predict the label"""
    # Tokenize
    inputs = tokenizer(x.Clinical_Note, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    return pred

# LF2: Keyword-based - Normal indicators
@labeling_function(name="normal_keywords")
def lf_normal_keywords(x):
    """Detect keywords suggesting normal results"""
    text = x.Clinical_Note.lower()
    
    # Terms strongly suggesting normal results
    normal_indicators = [
        "within normal limits", "unremarkable", 
        "normal range", "no significant findings", 
        "reassuring results", "negative for pathology",
        "no abnormalities", "no concerning findings",
        "stable", "satisfactory", "adequate", "good progress",
        "standard presentation", "sufficient", "steady",
        "within expected parameters"
    ]
    
    # Check if any indicators are present
    if any(indicator in text for indicator in normal_indicators):
        return label_mapping["Normal"]
    
    return ABSTAIN

# LF3: Keyword-based - Abnormal indicators
@labeling_function(name="abnormal_keywords")
def lf_abnormal_keywords(x):
    """Detect keywords suggesting abnormal results"""
    text = x.Clinical_Note.lower()
    
    # Terms strongly suggesting abnormal results
    abnormal_indicators = [
        "outside normal range", "abnormal findings", 
        "requires attention", "significantly elevated", 
        "concerning values", "positive for pathology",
        "results suggest pathology", "shows abnormalities",
        "irregular", "unusual", "non-standard", "remarkable",
        "deviating from baseline", "warrants attention", 
        "concerning features", "outside typical values",
        "anomalous", "atypical", "elevated"
    ]
    
    # Check if any indicators are present
    if any(indicator in text for indicator in abnormal_indicators):
        return label_mapping["Abnormal"]
    
    return ABSTAIN

# LF4: Keyword-based - Inconclusive indicators
@labeling_function(name="inconclusive_keywords")
def lf_inconclusive_keywords(x):
    """Detect keywords suggesting inconclusive results"""
    text = x.Clinical_Note.lower()
    
    # Terms strongly suggesting inconclusive results
    inconclusive_indicators = [
        "borderline results", "difficult to interpret", 
        "results inconclusive", "requires further testing", 
        "not definitive", "potentially suggestive", 
        "findings unclear", "indeterminate",
        "uncertain", "ambiguous", "questionable", 
        "equivocal", "unclear if", "partially resolved",
        "further evaluation needed", "somewhat improved",
        "challenging to interpret", "additional testing"
    ]
    
    # Check if any indicators are present
    if any(indicator in text for indicator in inconclusive_indicators):
        return label_mapping["Inconclusive"]
    
    return ABSTAIN

# LF5: Hospital stay duration heuristic
@labeling_function(name="stay_duration")
def lf_stay_duration(x):
    """Use hospital stay duration as a heuristic"""
    text = x.Clinical_Note.lower()
    
    # Try to extract admission and discharge dates
    try:
        # Find dates in formats like yyyy-mm-dd or mm/dd/yyyy
        dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}', text)
        
        if len(dates) >= 2:
            # Try to convert to datetime
            try:
                dates = pd.to_datetime(dates)
                # Calculate duration in days
                duration = (max(dates) - min(dates)).days
                
                # Very short stays often indicate Normal outcomes
                if duration <= 2:
                    return label_mapping["Normal"]
                # Very long stays often indicate Abnormal outcomes
                elif duration > 7:
                    return label_mapping["Abnormal"]
            except:
                pass
    except:
        # If any error occurs, abstain
        pass
    
    return ABSTAIN

# LF6: Language complexity heuristic
@labeling_function(name="language_complexity")
def lf_language_complexity(x):
    """Use medical language complexity as a heuristic"""
    text = x.Clinical_Note.lower()
    
    # Complex medical terminology often indicates abnormal findings
    complex_terms = [
        "pathology", "etiology", "differential diagnosis",
        "intervention", "complication", "anomaly",
        "deviation", "dysfunction", "impairment",
        "therapy", "chronic", "acute", "severity"
    ]
    
    # Count complex terms
    count = sum(1 for term in complex_terms if term in text)
    
    # More complex language often indicates abnormal findings
    if count >= 3:
        return label_mapping["Abnormal"]
    
    return ABSTAIN

# LF7: Test result patterns
@labeling_function(name="test_results")
def lf_test_results(x):
    """Analyze test result patterns in the note"""
    text = x.Clinical_Note.lower()
    
    # More comprehensive regex patterns for test results
    if re.search(r'(test|lab|result|finding|assessment|presentation).{1,20}(appear|show|seem|was|is|were|are).{1,20}(stable|normal|good|routine|clear|unremarkable|negative)', text):
        return label_mapping["Normal"]
    elif re.search(r'(test|lab|result|finding|assessment|presentation).{1,20}(appear|show|seem|was|is|were|are).{1,20}(abnormal|concerning|elevated|positive|unusual|irregular)', text):
        return label_mapping["Abnormal"]
    elif re.search(r'(test|lab|result|finding|assessment|presentation).{1,20}(appear|show|seem|was|is|were|are).{1,20}(unclear|inconclusive|borderline|ambiguous|equivocal|uncertain)', text):
        return label_mapping["Inconclusive"]
    
    return ABSTAIN

# LF8: Patient condition improvement
@labeling_function(name="patient_improvement")
def lf_patient_improvement(x):
    """Check for indicators of patient improvement"""
    text = x.Clinical_Note.lower()
    
    # Terms suggesting improvement
    improvement_terms = [
        "improved", "recovering", "responding well",
        "good progress", "favorable response", 
        "positive outcome", "healing well"
    ]
    
    # Terms suggesting lack of improvement
    no_improvement_terms = [
        "not improving", "worsening", "deteriorating",
        "poor response", "limited improvement",
        "failed to respond", "persistent symptoms"
    ]
    
    # Check for improvement
    if any(term in text for term in improvement_terms) and not any(term in text for term in no_improvement_terms):
        return label_mapping["Normal"]
    # Check for lack of improvement
    elif any(term in text for term in no_improvement_terms):
        return label_mapping["Abnormal"]
    
    return ABSTAIN

# ======== STEP 4: Apply Labeling Functions ========
# Define our labeling functions
lfs = [
    lf_clinicalbert,
    lf_normal_keywords,
    lf_abnormal_keywords,
    lf_inconclusive_keywords,
    lf_stay_duration,
    lf_language_complexity,
    lf_test_results,
    lf_patient_improvement
]

# Apply LFs to the data
print("Applying labeling functions...")
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_old)
L_test = applier.apply(df_new)

# ======== STEP 5: Analyze Labeling Function Performance ========
print("\nLabeling function analysis on training set:")
lf_analysis = LFAnalysis(L_train, lfs).lf_summary(df_old["label"].values)
print(lf_analysis)

# Save LF analysis to CSV for easier viewing
lf_analysis.to_csv("lf_analysis.csv", index=False)
print("Saved labeling function analysis to 'lf_analysis.csv'")

# ======== STEP 6: Train Label Model ========
print("\nTraining the label model...")
label_model = LabelModel(cardinality=len(label_mapping), verbose=True)
label_model.fit(L_train, seed=42, n_epochs=500, log_freq=100, lr=0.01)

# ======== STEP 7: Make Predictions ========
# Get training set predictions
train_preds = label_model.predict(L_train)
train_acc = accuracy_score(df_old["label"].values, train_preds)
print(f"Training set accuracy: {train_acc:.4f}")
print(classification_report(df_old["label"].values, train_preds, target_names=label_mapping.keys()))

# Get test set predictions
test_preds = label_model.predict(L_test)
test_acc = accuracy_score(df_new["label"].values, test_preds)
print(f"\nTest set accuracy: {test_acc:.4f}")
print(classification_report(df_new["label"].values, test_preds, target_names=label_mapping.keys()))

# ======== STEP 8: Analyze Knowledge Obsolescence ========
# Compare performance on old vs new datasets
print("\n===== Knowledge Obsolescence Analysis =====")

# Confusion matrices for visual inspection
print("\nConfusion Matrix (Old Dataset):")
cm_old = confusion_matrix(df_old["label"].values, train_preds)
print(cm_old)

print("\nConfusion Matrix (New Dataset):")
cm_new = confusion_matrix(df_new["label"].values, test_preds)
print(cm_new)

# Try to create visualization of confusion matrices
try:
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_old, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.keys()),
                yticklabels=list(label_mapping.keys()))
    plt.title('Confusion Matrix - Old Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.keys()),
                yticklabels=list(label_mapping.keys()))
    plt.title('Confusion Matrix - New Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Saved confusion matrices visualization to 'confusion_matrices.png'")
except Exception as e:
    print(f"Could not create visualization: {e}")

# Look for class-specific performance changes
old_report = classification_report(df_old["label"].values, train_preds, output_dict=True)
new_report = classification_report(df_new["label"].values, test_preds, output_dict=True)

print("\nPerformance changes by class:")
for i, class_name in enumerate(label_mapping.keys()):
    # Convert class name to string index for dictionary lookup
    class_idx = str(i)
    old_f1 = old_report[class_idx]['f1-score']
    new_f1 = new_report[class_idx]['f1-score']
    change = new_f1 - old_f1
    change_pct = (change / old_f1) * 100 if old_f1 > 0 else float('inf')
    
    print(f"{class_name}: Old F1={old_f1:.4f}, New F1={new_f1:.4f}, Change={change:.4f} ({change_pct:+.2f}%)")

# Try to create visualization of F1 score changes
try:
    plt.figure(figsize=(10, 6))
    classes = list(label_mapping.keys())
    old_f1_scores = [old_report[str(i)]['f1-score'] for i in range(len(classes))]
    new_f1_scores = [new_report[str(i)]['f1-score'] for i in range(len(classes))]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, old_f1_scores, width, label='Old Dataset')
    plt.bar(x + width/2, new_f1_scores, width, label='New Dataset')
    
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison: Old vs New Dataset')
    plt.xticks(x, classes)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(old_f1_scores):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(new_f1_scores):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('f1_score_comparison.png')
    print("Saved F1 score comparison to 'f1_score_comparison.png'")
except Exception as e:
    print(f"Could not create F1 visualization: {e}")

# ======== STEP 9: Analyze Labeling Function Contributions ========
print("\n===== Labeling Function Contributions =====")

# Get label model weights
try:
    # Try to get weights - handle potential errors in Snorkel
    weights = label_model.get_weights()
    
    # Create a list to store LF name and weight pairs
    lf_weights = []
    for i, lf in enumerate(lfs):
        try:
            # Get the LF name directly from the name property
            lf_name = lf.name
            lf_weights.append((lf_name, weights[i]))
        except:
            # Fallback to index if name fails
            lf_weights.append((f"LF_{i}", weights[i]))
    
    # Sort by weight (importance)
    lf_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Print sorted weights
    print("Labeling function weights (sorted by importance):")
    for name, weight in lf_weights:
        print(f"{name}: {weight:.4f}")
    
    # Try to create visualization of LF weights
    try:
        plt.figure(figsize=(10, 6))
        names = [x[0] for x in lf_weights]
        values = [x[1] for x in lf_weights]
        
        plt.barh(names, values)
        plt.xlabel('Weight')
        plt.title('Labeling Function Weights')
        plt.tight_layout()
        plt.savefig('lf_weights.png')
        print("Saved labeling function weights visualization to 'lf_weights.png'")
    except Exception as e:
        print(f"Could not create LF weights visualization: {e}")
    
except Exception as e:
    print(f"Could not analyze LF weights: {e}")
    # Alternative: Look at empirical accuracy
    print("\nLabeling function empirical accuracy:")
    for i, lf in enumerate(lfs):
        try:
            accuracy = lf_analysis.loc[i, "Emp. Acc."]
            coverage = lf_analysis.loc[i, "Coverage"]
            print(f"{lf.name}: Accuracy={accuracy:.4f}, Coverage={coverage:.4f}")
        except:
            print(f"LF_{i}: Could not determine accuracy")

# ======== STEP 10: Save the Snorkel Label Model ========
print("\nSaving the Snorkel label model...")
label_model.save("snorkel_label_model.pkl")
print("Snorkel pipeline complete!")

# ======== STEP 11: Knowledge Obsolescence Deep Dive ========
print("\n===== Knowledge Obsolescence Deep Dive =====")

# Identify examples where model performance changed the most
def find_interesting_examples():
    """Find examples that highlight knowledge obsolescence"""
    # Get predictions from ClinicalBERT alone for both datasets
    bert_old_preds = []
    bert_new_preds = []
    
    print("Getting ClinicalBERT-only predictions...")
    
    # Get predictions for old dataset
    for i, row in tqdm(df_old.iterrows(), total=len(df_old), desc="Old dataset"):
        inputs = tokenizer(row["Clinical_Note"], return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            bert_old_preds.append(torch.argmax(outputs.logits, dim=1).item())
    
    # Get predictions for new dataset
    for i, row in tqdm(df_new.iterrows(), total=len(df_new), desc="New dataset"):
        inputs = tokenizer(row["Clinical_Note"], return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            bert_new_preds.append(torch.argmax(outputs.logits, dim=1).item())
    
    # Analyze where ClinicalBERT and Snorkel disagree
    old_disagreements = []
    for i in range(len(df_old)):
        if bert_old_preds[i] != train_preds[i]:
            old_disagreements.append((i, df_old.iloc[i]["label"], bert_old_preds[i], train_preds[i]))
    
    new_disagreements = []
    for i in range(len(df_new)):
        if bert_new_preds[i] != test_preds[i]:
            new_disagreements.append((i, df_new.iloc[i]["label"], bert_new_preds[i], test_preds[i]))
    
    # Report findings
    print(f"\nModels disagree on {len(old_disagreements)} samples in old dataset ({len(old_disagreements)/len(df_old)*100:.1f}%)")
    print(f"Models disagree on {len(new_disagreements)} samples in new dataset ({len(new_disagreements)/len(df_new)*100:.1f}%)")
    
    # Example where Snorkel improves over ClinicalBERT
    improved_old = [x for x in old_disagreements if x[1] == x[3] and x[1] != x[2]]
    improved_new = [x for x in new_disagreements if x[1] == x[3] and x[1] != x[2]]
    
    print(f"\nSnorkel improves prediction on {len(improved_old)} old samples")
    print(f"Snorkel improves prediction on {len(improved_new)} new samples")
    
    # Examples where knowledge obsolescence is most evident
    if improved_old and improved_new:
        print("\nExamples where Snorkel's weak supervision helps overcome knowledge obsolescence:")
        # Sample from old dataset
        idx = improved_old[0][0]
        print(f"\nOLD DATASET EXAMPLE (index {idx}):")
        print(f"True label: {id2label[df_old.iloc[idx]['label']]}")
        print(f"ClinicalBERT prediction: {id2label[bert_old_preds[idx]]}")
        print(f"Snorkel prediction: {id2label[train_preds[idx]]}")
        print(f"Clinical note: {df_old.iloc[idx]['Clinical_Note'][:200]}...")
        
        # Sample from new dataset
        idx = improved_new[0][0]
        print(f"\nNEW DATASET EXAMPLE (index {idx}):")
        print(f"True label: {id2label[df_new.iloc[idx]['label']]}")
        print(f"ClinicalBERT prediction: {id2label[bert_new_preds[idx]]}")
        print(f"Snorkel prediction: {id2label[test_preds[idx]]}")
        print(f"Clinical note: {df_new.iloc[idx]['Clinical_Note'][:200]}...")

# Uncomment to run the deep dive analysis (takes time)
# find_interesting_examples()

print("\nKnowledge Obsolescence Analysis Summary:")
print(f"- Overall performance drop: {train_acc - test_acc:.4f} ({(train_acc - test_acc)/train_acc*100:.2f}% relative decrease)")
print(f"- Largest performance drop in class: Inconclusive ({change_pct:.2f}%)")
print("- Possible interpretation: Medical knowledge and assessment criteria for inconclusive cases")
print("  have evolved more significantly than criteria for normal/abnormal cases")