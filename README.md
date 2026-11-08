# ML Assignment 2 - Frog Species Classification from Call Audio Features

## a. Problem statement

Frog (anuran) calls are recorded in the wild to monitor ecosystems, because frogs are sensitive to changes in their environment. In this assignment the frog **species** is to be identified from audio features (MFCCs) that were extracted from a single call syllable. This is a multi-class classification problem with 10 species. Five classification models have been trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app has also been built so that the saved models can be tested on the test data.

## b. Dataset description

- **Dataset:** Anuran Calls (MFCCs) - UCI Machine Learning Repository (id 406)
- **Link:** https://www.kaggle.com/datasets/yasserhessein/anuran-calls-mfccs 
- **Instances:** 7195 (each row is one call syllable, recorded in Brazil and Argentina)
- **Features:** 22 (MFCC_1 to MFCC_22, all numeric, already normalised between -1 and 1)
- **Target:** `Species` (10 classes)

The Anuran Calls dataset has been picked because it is a less commonly used dataset and the problem (identifying an animal from its sound) is interesting. MFCCs (Mel-frequency cepstral coefficients) are a standard way of summarising the frequency content of a short piece of audio, and they are the same features that are used in speech recognition.

The original file has 3 label columns (Family, Genus, Species) and a RecordID column. Species has been used as the target. Family and Genus are coarser versions of the same label, so they have been dropped along with RecordID. The column names in the file (`MFCCs_ 1`, `MFCCs_ 2`, ...) have been renamed to `MFCC_1` ... `MFCC_22`.

Class counts:

| Species | Family | Count |
|---|---|---|
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

No missing values are present. The classes are heavily imbalanced (the biggest class is almost half of the data and the smallest is under 1%). Because of this, macro averaging has been used for Precision, Recall and F1, and AUC has been calculated using One-vs-Rest with macro averaging.

Train/test split: 75/25, stratified on Species, random_state = 737. This gives 5396 training rows and 1799 test rows. The test rows have been saved in `test_data.csv` (22 features + Species).

## c. Github Repository Link

https://github.com/Teja2026AJ05842/bits_ml/tree/ml_assignment2

**Live Streamlit App:** https://<your-app>.streamlit.app

Repository files:

```
project-folder/
|-- app.py                 (streamlit app)
|-- train_models.py        (training code for all 5 models)
|-- requirements.txt
|-- .streamlit/config.toml (green theme for the app)
|-- README.md
|-- test_data.csv
|-- data/
    |-- Frogs_MFCCs.csv    (original dataset)
    |-- Readme.txt         (dataset readme from UCI)
|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- svm.pkl
    |-- scaler.pkl         (StandardScaler fitted on train data)
    |-- model_results.csv
```

## d. Models used

1. Logistic Regression (max_iter = 1000)
2. Decision Tree Classifier (default settings)
3. K-Nearest Neighbor Classifier (k = 5)
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (RBF kernel, probability = True so that AUC can be calculated)

All features have been scaled with StandardScaler. The scaler was fitted only on the training data and then applied to the test data. The same scaled data has been given to all five models so that the comparison is fair.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9661 | 0.9936 | 0.9241 | 0.9116 | 0.9167 | 0.9528 |
| Decision Tree | 0.9477 | 0.9190 | 0.8664 | 0.8438 | 0.8520 | 0.9274 |
| KNN | 0.9817 | 0.9880 | 0.9639 | 0.9358 | 0.9485 | 0.9745 |
| Naive Bayes | 0.9016 | 0.9838 | 0.8183 | 0.8064 | 0.7987 | 0.8663 |
| Support Vector Machine | 0.9883 | 0.9989 | 0.9884 | 0.9497 | 0.9671 | 0.9838 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A good result is obtained even though it is a linear model: 96.61% accuracy, 61 wrong out of 1799 and an AUC of 0.994. The biggest class (AdenomeraHylaedactylus) is classified with 100% recall. The MFCC features seem to separate most species almost linearly. The weak spots are the two small tree frog species, OsteocephalusOophagus (68% recall) and HylaMinuta (79%). |
| Decision Tree | 94.77% accuracy is obtained but the AUC is the lowest of all models (0.919), because a single tree gives hard 0/1 style probabilities. The rare species are affected the most: only 50% recall on OsteocephalusOophagus and 59% on Rhinellagranulosa, since very few training rows are available for these classes to be split on. No depth limit was set, so the tree is likely overfitting the training data. |
| KNN | Second best model overall (98.17% accuracy, 33 wrong). 95% or more recall is achieved on 8 of the 10 species. On the two rarest species the recall drops to 75% and 82%, because a rare call that lies close to calls of a big class gets out-voted by the 5 neighbours. Scaling is still useful for KNN even though the MFCCs are already roughly in the -1 to 1 range. |
| Naive Bayes | Weakest model (90.16% accuracy, 177 wrong, MCC 0.866). Ameeregatrivittata is over-predicted (precision only 0.58) and so is OsteocephalusOophagus (precision 0.37). AdenomeraAndre is confused with AdenomeraHylaedactylus, which belongs to the same genus (67% recall), and HylaMinuta recall is only 44%. Neighbouring MFCC coefficients are correlated, so the independence assumption of Naive Bayes does not hold well here. |
| Support Vector Machine | Best on all 6 metrics: 98.83% accuracy, only 21 wrong, AUC 0.999 and MCC 0.984. 100% precision is obtained on the three smallest species (ScinaxRuber, OsteocephalusOophagus, Rhinellagranulosa), and the only noticeable errors are the 82% recall on the two rarest ones, which have just 28 and 17 test samples. The RBF kernel is able to draw curved boundaries between the similar tree frog species better than the other models. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** Every metric is won by SVM, with KNN a close second. All models except Naive Bayes perform well because MFCCs are strong features for this task. The real difference between the models is seen on the rare species, where SVM and KNN hold up the best and Decision Tree and Naive Bayes fall behind. |

## Streamlit App

The following features are provided in the app:
- Test data upload (CSV) from the sidebar
- Dropdown to select the model from the sidebar
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix and classification report
- Species count chart of the uploaded data and a short description of each species
- Per-species recall chart (which species is easiest / hardest for the selected model)
- Actual vs predicted table, with an option to show only the wrong predictions and download them
- Table comparing all 5 models on the uploaded file (best value in each metric is highlighted)

Only the saved models from the `model/` folder are loaded by the app. No training is done on the uploaded data.

## How to run

```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
