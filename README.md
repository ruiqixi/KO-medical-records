# KO Problem in Medical Field

This project investigates **Knowledge Obsolescence (KO)** in the medical domain using clinical notes. It leverages advanced NLP models and weak supervision techniques to analyze shifts in classification performance over time and uncover signs of outdated medical knowledge.

---
## Installation & Execution

To reproduce this project, follow the steps below to set up your environment and run the full pipeline, including ClinicalBERT training, Snorkel labeling, and domain adaptation.

### Prerequisites

Ensure you have **Python 3.8+** installed. It's recommended to use a virtual environment (e.g., `venv` or `conda`).

### Required Packages

Install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
### 🔧 Running the Pipeline

#### 1️⃣ Train the Base ClinicalBERT Model

```bash
python train_clinicalbert.py
```

This command trains ClinicalBERT on the **old dataset** (`preprocessed_clinical_notes_old.csv`) and evaluates it on the **new dataset** (`preprocessed_clinical_notes_new.csv`).  
The fine-tuned model will be saved in the `clinicalbert_for_snorkel/` directory.

---

#### 2️⃣ Apply Snorkel Weak Supervision

```bash
python snorkel_implement.py
```

This step runs **12 labeling functions (LFs)**, applies the Snorkel label model to both datasets, and evaluates the model using weak supervision.  
It also outputs label function analysis and confusion matrices.

---

#### 3️⃣ Apply Domain Adaptation

```bash
python domain_adaptation.py
```

---

## Datasets
- Metadata:
  | Dataset                         | # Rows | # Columns | Column Names |
  |---------------------------------|--------|-----------|--------------|
  | `Healthcare.csv`                | 55,500 | 15        | Name, Age, Gender, Blood Type, Medical Condition, Date of Admission, Doctor, Hospital, Insurance Provider, Billing Amount, Room Number, Admission Type, Discharge Date, Medication, Test Results |
  | `preprocessed_clinical_notes_old.csv` | 876    | 2         | Clinical_Note, Outcome |
  | `preprocessed_clinical_notes_new.csv` | 828    | 2         | Clinical_Note, Outcome |
- **`Healthcare.csv`**
   Original dataset containing raw clinical records.
- Example data item:
  Bobby JacksOn	30	Male	B-	Cancer	2024/1/31	Matthew Smith	Sons and Miller	Blue Cross	18856.28131	328	Urgent	2024/2/2	Paracetamol	Normal![image](https://github.com/user-attachments/assets/4089f24e-50f0-4c38-8f20-48cd66f57841)

- **`preprocessed_clinical_notes_new.csv`**
  Preprocessed version of newer clinical notes (from 2022 to 2024).
- Example data item:
  77/Male with Hypertension. partially resolved. Admitted 2023-07-10. Labs: assessed. within expected parameters. Plan discussed with patient. Tx: Aspirin. requires monitoring. DC 2023-07-14.	Abnormal![image](https://github.com/user-attachments/assets/fa146b78-f206-42a8-ac8e-5252fd4eba30)

- **`preprocessed_clinical_notes_old.csv`**
  Preprocessed version of recent clinical notes (from 2019 to 2021).
- Example data item:
  patient is a 58-year-old Male who presented with Asthma. difficult to determine at Obrien Group after admission on 2019-07-08. Plan discussed with patient. Test results: assessed. concerning features. Treatment included Lipitor as prescribed by Dr. Derek Martin. further evaluation needed. Patient was discharged on 2019-07-17.	Inconclusive![image](https://github.com/user-attachments/assets/860db2ce-80cc-44c0-8d03-d4ca55e44e32)

---

## Code Files

- **`data_preprocessing.py`**  
  ➤ **Output:** Preprocesses raw data and outputs the two cleaned datasets (`preprocessed_clinical_notes_old.csv`, `preprocessed_clinical_notes_new.csv`).

- **`train_clinicalbert.py`**  
  Trains a ClinicalBERT model for multi-class classification on the preprocessed data.  
  ➤ **Output:** `clinicalbert_for_snorkel`
    **Best Validation Metrics:**
    - Accuracy: `0.8258`
    - F1 Score: `0.8255`
    
    **Test Metrics:**
    - Accuracy: `0.7415`
    - F1 Score: `0.7416`
    
    **Training Parameters:**
    - Batch Size: `16`
    - Learning Rate: `2e-05`
    - Max Sequence Length: `86`
    - Weight Decay: `0.01`
    - Epochs: `8`

- **`snorkel_implement.py`**  
  Implements Snorkel weak supervision pipeline, generates label model, evaluates KO effects, and visualizes results.
  ➤ **Output:**
    **Training Set Performance:**
    - Accuracy: `0.8973`
    
    **Test Set Performance:**
    - Accuracy: `0.7464`
    **Key Visual Outputs:**
    - `confusion_matrices.png` – Side-by-side confusion matrices for old and new datasets.
    - `f1_score_comparison.png` – Visualization of F1 score degradation due to KO.
      
  - **`domain_adaptation.py`**  
  Implements domain adaptation technique to reach higher performance of new dataset and visualizes results.
  ➤ **Output:**
    **Training Set Performance:**
    - Accuracy: `0.9073`
    
    **Test Set Performance:**
    - Accuracy: `0.7952`
    **Key Visual Outputs:**
    - `model_comparison.png` – Comparing results of different ensembled models
---

## Evaluation Summary & KO Analysis

- **KO Problem Observed**:  
  The base ClinicalBERT model, trained on 2019–2021 data, dropped to **71.7% accuracy** when evaluated on the 2022–2024 dataset—demonstrating clear signs of **knowledge obsolescence**.

- **Biggest Performance Dip**:  
  The _Inconclusive_ class saw the largest drop, likely due to subtle shifts in language and medical documentation over time.

- **Snorkel Attempt (Weak Supervision)**:  
  Implementing weak supervision with Snorkel led to a moderate improvement (**74.4% accuracy**), achieved **without any additional manual labeling**.

- **Snorkel Limitations**:
  - The total time span was only **six years**, with a three-year overlap between old and new datasets.
  - **Minimal language drift** in terminology made it hard for labeling functions (LFs) to capture meaningful changes.

- **Domain Adaptation Impact**:
  - Fine-tuning ClinicalBERT with **new labeled data** and **text augmentation** (e.g., synonym replacement, date shifting) significantly improved model adaptability.
  - The **ensemble model** (Base + New Data + Hybrid) achieved the best result:  
    **79.5% accuracy**, recovering **89.9%** of original model performance.

- **Why Domain Adaptation Succeeded**:
  - It retrained the model’s internal embeddings to learn **evolving phrasing**, **abbreviations**, and **documentation styles**.
  - Hybrid and ensemble strategies allowed the model to **retain old knowledge** while aligning with **modern data**—directly addressing the KO challenge.




---

## Project Summary

This project highlights the impact of **knowledge aging in clinical NLP**, proposes a weakly-supervised labeling strategy with **Snorkel**, and evaluates robustness of ClinicalBERT under temporal domain shift. It provides a framework to study knowledge obsolescence in sensitive fields like medicine.
