# SWYNEX-Model-or-API-Integration

**Project Title:** Automated Daily Vegetable Price & Stock Demand Classifier (ML Prototype)  
**Author:** Krishna Vaishy  

---

### Overview
This prototype implements the narrow-AI solution designed in Task 1 using **Python**, **pandas**, and **scikit-learn**. It trains a lightweight `RandomForestClassifier` pipeline on structured daily retail vegetable sales and wholesale (*mandi*) pricing records to classify morning procurement decisions into three tiers: `High_Restock`, `Maintain`, or `Reduce`.

No external paid APIs or secret keys are required—the model runs completely locally on CPU.

---

### How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt