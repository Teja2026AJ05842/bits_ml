# ML Assignment 2 - Frog Species Classification from Call Audio Features

## a. Problem statement

Frog (anuran) calls are recorded in the wild to monitor ecosystems, because frogs are sensitive to changes in their environment. In this assignment, the frog **species** is identified from audio features called MFCCs that were extracted from a single call syllable.

This is a multi-class classification problem with 10 species. Five classification models have been trained on the same dataset and compared using Accuracy, AUC, Precision, Recall, F1 and MCC.

A Streamlit app has also been built so that the saved models can be tested using the test data.

---

## b. Dataset description

- **Dataset:** Anuran Calls (MFCCs) - UCI Machine Learning Repository (id 406)
- **Link:** https://www.kaggle.com/datasets/yasserhessein/anuran-calls-mfccs
- **Instances:** 7195
- **Features:** 22 (`MFCC_1` to `MFCC_22`)
- **Target:** `Species` (10 classes)

Each row represents one frog call syllable recorded in Brazil and Argentina.

The Anuran Calls dataset was selected because it is a less commonly used classification dataset and contains audio-based features. MFCCs (Mel-frequency cepstral coefficients) are used to represent the frequency information present in an audio signal.

The original dataset contains three label columns: `Family`, `Genus` and `Species`, along with a `RecordID` column.

For this assignment, `Species` has been used as the target. `Family`, `Genus` and `RecordID` were removed before training the models.

The original feature names such as `MFCCs_ 1`, `MFCCs_ 2`, etc. were renamed to `MFCC_1`, `MFCC_2`, up to `MFCC_22`.

### Class counts

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

The class distribution is imbalanced. The largest class contains 3478 samples while the smallest class contains only 68 samples.

Because of this, macro averaging has been used for Precision, Recall and F1. AUC has been calculated using One-vs-Rest with macro averaging.

The dataset was divided into training and testing sets using a **75/25 split**. Stratification was applied on the `Species` column with:

`random_state = 737`

This resulted in:

- **Training rows:** 5396
- **Testing rows:** 1799

The testing rows were saved in `test_data.csv`, containing the 22 MFCC features and the `Species` target column.

---

## c. Github Repository Link

https://github.com/Teja2026AJ05842/bits_ml/tree/ml_assignment2

**Note:** The assignment code is available on the `ml_assignment2` branch of the repository. The Streamlit application is also deployed from this branch.

**Live Streamlit App:**  
https://teja2026aj05842-bits-ml-app-ml-assignment2-pmieuc.streamlit.app/

### Repository files

```text
project-folder/

|-- app.py                  (Streamlit app)

|-- train_models.py         (training code for all 5 models)

|-- requirements.txt

|-- .streamlit/
    |-- config.toml         (green theme for the app)

|-- README.md

|-- test_data.csv

|-- data/
    |-- Frogs_MFCCs.csv     (original dataset)
    |-- Readme.txt          (dataset readme from UCI)

|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- svm.pkl
    |-- scaler.pkl          (StandardScaler fitted on training data)
    |-- model_results.csv
```

---

## d. Models used

The following five classification models were implemented:

1. Logistic Regression (`max_iter = 1000`)
2. Decision Tree Classifier (default settings)
3. K-Nearest Neighbor Classifier (`k = 5`)
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (RBF kernel, `probability = True`)

`probability = True` was used for SVM so that probability values could be obtained for AUC calculation.

All features were scaled using `StandardScaler`.

The scaler was fitted only on the training data and then applied to the test data. The same scaled data was used for all five models.

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

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved **96.61% accuracy** and an AUC of **0.9936**. It made 61 incorrect predictions out of 1799 test samples. The model achieved 100% recall for AdenomeraHylaedactylus. Lower recall was seen for OsteocephalusOophagus and HylaMinuta species targets. |
| **Decision Tree** | Decision Tree achieved **94.77% accuracy**. Its AUC of **0.9190** was the lowest among the five models. Recall was lower for some of the smaller classes, including OsteocephalusOophagus and Rhinellagranulosa. The classifier was used with its default settings and no maximum depth was specified. |
| **KNN** | KNN achieved **98.17% accuracy** and made 33 incorrect predictions. It achieved high recall for most of the 10 species. The recall was lower for some of the rare classes because they contain fewer samples. The value of `k` was set to 5 and the input features were scaled before training. |
| **Naive Bayes** | Naive Bayes achieved **90.16% accuracy**, which was the lowest accuracy among the five models. Its MCC score was **0.8663**. Some confusion was seen between species such as AdenomeraAndre and AdenomeraHylaedactylus. HylaMinuta also had lower recall compared to the other models. |
| **Support Vector Machine** | Support Vector Machine achieved **98.83% accuracy**, an AUC of **0.9989** and an MCC of **0.9838**. It made only 21 incorrect predictions out of 1799 test samples. It also achieved 100% precision for ScinaxRuber, OsteocephalusOophagus and Rhinellagranulosa. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** achieved the highest Accuracy, AUC, Precision, Recall, F1 and MCC among the five models. KNN was the second-best model based on the overall results. |

---

## Streamlit App

The Streamlit application provides the following features:

- Upload test data in CSV format from the sidebar
- Option to use the `test_data.csv` included in the repo directly (checkbox in the sidebar)
- Select a machine learning model using a dropdown
- Display Accuracy, AUC, Precision, Recall, F1 and MCC
- Display the confusion matrix
- Display the classification report
- Show the species count distribution of the uploaded dataset
- Show the recall for each species
- Display actual and predicted values
- Option to display only incorrect predictions
- Option to download incorrect predictions
- Compare all five models using the uploaded test data
- Highlight the best value for each evaluation metric

Only the saved models from the `model/` folder are loaded by the Streamlit application.

No training is performed on the uploaded test data.

---

## How to run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```