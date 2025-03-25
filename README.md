# KO Problem in Medical Field

This project investigates **Knowledge Obsolescence (KO)** in the medical domain using clinical notes. It leverages advanced NLP models and weak supervision techniques to analyze shifts in classification performance over time and uncover signs of outdated medical knowledge.

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
    - Accuracy: `0.8562`
    - F1-Score (macro avg): `0.86`
    
    **Test Set Performance:**
    - Accuracy: `0.7271`
    - F1-Score (macro avg): `0.73`
    **Key Visual Outputs:**
    - `confusion_matrices.png` – Side-by-side confusion matrices for old and new datasets.
    - `lf_weights.png` – Contribution weights of labeling functions.
    - `f1_score_comparison.png` – Visualization of F1 score degradation due to KO.
---

## Knowledge Obsolescence Insights

- **Overall F1 score drop:** `-0.1291` (≈15.08% relative decrease)
- **Greatest impact on class:** `Inconclusive`  
  → F1 Score dropped from `0.8237` → `0.6453`  
  → Relative change: `-21.66%`

**Interpretation:**  
The handling of *inconclusive* cases has evolved more drastically, suggesting these are most sensitive to outdated clinical assessment criteria.

---

## Project Summary

This project highlights the impact of **knowledge aging in clinical NLP**, proposes a weakly-supervised labeling strategy with **Snorkel**, and evaluates robustness of ClinicalBERT under temporal domain shift. It provides a framework to study knowledge obsolescence in sensitive fields like medicine.
