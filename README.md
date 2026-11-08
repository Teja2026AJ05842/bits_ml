# ML Assignment 2 - Frog Species Classification from Call Audio Features

**Course:** S1-26_AIMLZG565  
**Name:** Sai Teja Koneru  
**Student ID:** 2026AJ05842  

---

## GitHub Repository

https://github.com/Teja2026AJ05842/bits_ml/tree/ml_assignment2

**Branch:** `ml_assignment2`

---

## Live Streamlit App

https://teja2026aj05842-bits-ml-app-ml-assignment2-pmieuc.streamlit.app/

---

## a. Problem Statement

Frog (anuran) calls are recorded in the wild to monitor ecosystems, because frogs are sensitive to changes in their environment.

In this assignment, the frog **species** is identified from audio features called MFCCs that were extracted from a single call syllable.

This is a multi-class classification problem with 10 species. Five classification models have been trained on the same data and compared using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

A Streamlit app has also been built so that the saved models can be tested on the test data.

---

## b. Dataset Description

- **Dataset:** Anuran Calls (MFCCs) - UCI Machine Learning Repository (id 406)
- **Link:** https://www.kaggle.com/datasets/yasserhessein/anuran-calls-mfccs
- **Instances:** 7195
- **Features:** 22 (`MFCC_1` to `MFCC_22`)
- **Target:** `Species` (10 classes)

Each row represents one frog call syllable recorded in Brazil and Argentina.

The Anuran Calls dataset was selected because it satisfies the assignment requirements and provides an interesting use case of identifying an animal from its sound.

MFCCs (Mel-frequency cepstral coefficients) are a standard way of summarising the frequency content of a short audio signal. Similar features are also used in speech recognition systems.

The original dataset contains three label columns:

- `Family`
- `Genus`
- `Species`

It also contains a `RecordID` column.

For this assignment, `Species` has been used as the target variable. `Family` and `Genus` are broader versions of the same label, so they were removed along with `RecordID`.

The original feature names such as `MFCCs_ 1`, `MFCCs_ 2`, etc. were renamed to `MFCC_1`, `MFCC_2`, up to `MFCC_22`.

### Class Counts

| Species | Family | Count |
|---|---|---:|
| AdenomeraHylaedactylus | Leptodactylidae | 3478 |
| HypsiboasCordobae | Hylidae | 1121 |
| AdenomeraAndre | Leptodactylidae | 672 |
| Ameeregatrivittata | Dendrobatidae | 542 |
| HypsiboasCinerascens | Hylidae | 472 |
| HylaMinuta | Hylidae | 310 |
| LeptodactylusFuscus | Leptodactylidae | 270 |
| ScinaxRuber | Hylidae | 148 |
| OsteocephalusOophagus | Hylidae | 114 |
| Rhinellagranulosa | Bufonidae | 68 |

There are no missing values in the dataset.

The classes are imbalanced. The largest class contains 3478 samples, while the smallest class contains only 68 samples.

Because of this imbalance, macro averaging has been used for:

- Precision
- Recall
- F1 Score

AUC has been calculated using **One-vs-Rest with macro averaging**.

### Train/Test Split

The dataset was divided using a **75/25 train-test split**.

- Stratified on: `Species`
- `random_state = 737`
- Training rows: **5396**
- Testing rows: **1799**

The test rows were saved in `test_data.csv` containing the 22 MFCC features and the `Species` target column.

---

## c. GitHub Repository Link

https://github.com/Teja2026AJ05842/bits_ml/tree/ml_assignment2

The assignment code is available on the `ml_assignment2` branch of the repository.

The Streamlit application is also deployed from this branch.

### Live Streamlit App

https://teja2026aj05842-bits-ml-app-ml-assignment2-pmieuc.streamlit.app/

### Repository Structure

```text
project-folder/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── Frogs_MFCCs.csv
│   └── Readme.txt
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── scaler.pkl
    └── model_results.csv
```

`app.py` contains the Streamlit application.

`train_models.py` contains the training and evaluation code for all five models.

`test_data.csv` contains the held-out test data used in the Streamlit application.

The `data/` folder contains the original dataset, while the `model/` folder contains the trained models, fitted scaler and model comparison results.

---

## d. Models Used

The following five classification models were implemented:

1. **Logistic Regression**
   - `max_iter = 1000`

2. **Decision Tree Classifier**
   - Default settings

3. **K-Nearest Neighbor Classifier**
   - `k = 5`

4. **Naive Bayes Classifier**
   - Gaussian Naive Bayes

5. **Support Vector Machine**
   - RBF kernel
   - `probability = True`

`probability = True` was enabled for SVM so that probability values could be used to calculate AUC.

All features were scaled using `StandardScaler`.

The scaler was fitted only on the training data and then applied to the test data.

The same scaled dataset was used for all five models.

---

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9661 | 0.9936 | 0.9241 | 0.9116 | 0.9167 | 0.9528 |
| Decision Tree | 0.9477 | 0.9190 | 0.8664 | 0.8438 | 0.8520 | 0.9274 |
| KNN | 0.9817 | 0.9880 | 0.9639 | 0.9358 | 0.9485 | 0.9745 |
| Naive Bayes | 0.9016 | 0.9838 | 0.8183 | 0.8064 | 0.7987 | 0.8663 |
| Support Vector Machine | 0.9883 | 0.9989 | 0.9884 | 0.9497 | 0.9671 | 0.9838 |

---

### Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved **96.61% accuracy** with an AUC of **0.9936**. It made 61 incorrect predictions out of 1799 test samples. The model achieved 100% recall for AdenomeraHylaedactylus. Lower recall was observed for OsteocephalusOophagus and HylaMinuta. Being a linear model, it finds it harder to separate these small classes from species with similar calls. |
| **Decision Tree** | Decision Tree achieved **94.77% accuracy**. Its AUC of **0.9190** was the lowest among the five models. Lower recall was observed for some of the smaller classes, mainly OsteocephalusOophagus and Rhinellagranulosa. The model was trained using the default Decision Tree settings. A single tree gives almost hard 0/1 probabilities, which is why its AUC is lower, and with no depth limit it may also be overfitting the training data. |
| **KNN** | KNN achieved **98.17% accuracy** and made 33 incorrect predictions. It achieved high recall for most of the 10 species. Some reduction in recall was seen for the classes with fewer samples. The value of `k` was set to 5 and the input features were scaled before training. The drop on small classes happens because a rare call close to calls of a bigger class gets outvoted by its 5 neighbours. |
| **Naive Bayes** | Naive Bayes achieved **90.16% accuracy**, which was the lowest accuracy among the five models. Its MCC score was **0.8663**. More confusion was observed between some species, including AdenomeraAndre and AdenomeraHylaedactylus. HylaMinuta also had lower recall compared to most of the other models. This is expected because Naive Bayes assumes the features are independent, while neighbouring MFCC values are correlated. |
| **Support Vector Machine** | Support Vector Machine achieved **98.83% accuracy**, an AUC of **0.9989** and an MCC of **0.9838**. It made 21 incorrect predictions out of 1799 test samples. It also achieved 100% precision for ScinaxRuber, OsteocephalusOophagus and Rhinellagranulosa. The RBF kernel was able to separate species with similar calls better than the other models. |
| **Overall, Winner for the Dataset** | **Support Vector Machine** achieved the highest Accuracy, AUC, Precision, Recall, F1 and MCC among the five models. KNN achieved the second-highest overall results. |

---

## Streamlit App

The Streamlit application provides the following features:

- Test data upload in CSV format (at the top of the page)
- **Autofill Test Data** button to load the `test_data.csv` from the repository directly
- Dropdown to select the machine learning model (placed above the results)
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Species count chart for the uploaded data
- Short description of each species
- Per-species recall chart
- Family check: species accuracy vs family accuracy, and how many wrong predictions still have the correct family
- Actual vs predicted results table
- Option to display only incorrect predictions
- Option to download the predictions as CSV
- Call fingerprint: line chart of the average MFCC values for selected species
- Check one call: a random test call predicted by all five models side by side
- Comparison table for all five models
- Best value in each metric highlighted in the comparison table

Only the saved models from the `model/` folder are loaded by the Streamlit app.

No model training is performed on the uploaded test data.

---

## BITS Virtual Lab Execution

The application was also executed locally on the **BITS Prayogshala Virtual Machine**.

The Streamlit application was run using:

```bash
streamlit run app.py
```

The BITS Virtual Lab screenshot is included in the final assignment PDF as proof of execution.

---

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Train and save the five models:

```bash
python train_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```