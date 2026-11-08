# ML Assignment 2 - Anuran Calls (MFCCs) - frog species classification
# trains the 5 models, prints the metrics and saves everything in model/ folder

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

SEED = 737
TEST_SIZE = 0.25
K_NEIGHBORS = 5

# load data (downloaded from kaggle / UCI id 406)
df = pd.read_csv('data/Frogs_MFCCs.csv')
print('Raw shape:', df.shape)
print('Missing values:', df.isnull().sum().sum())

# column names in the file look like "MFCCs_ 1", rename them to MFCC_1 ... MFCC_22
df.columns = [c.replace('MFCCs_', 'MFCC_').replace(' ', '') for c in df.columns]

# dataset has 3 label columns (Family, Genus, Species). I am using Species as the target.
# RecordID is just an id so dropping it along with Family and Genus
df = df.drop(columns=['Family', 'Genus', 'RecordID'])
feature_cols = [c for c in df.columns if c != 'Species']
print('Features:', len(feature_cols))
print(df['Species'].value_counts())

X = df[feature_cols]
y = df['Species']

# 80/20 split, stratify because some species have very few samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
print('Train:', X_train.shape, 'Test:', X_test.shape)

# save test data for the streamlit app
test_df = X_test.copy()
test_df['Species'] = y_test
test_df.to_csv('test_data.csv', index=False)

# scaling (MFCCs are already roughly in -1..1 but LR/KNN/SVM still work better with scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=K_NEIGHBORS),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, random_state=SEED),
}

file_names = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Support Vector Machine': 'svm.pkl',
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # 10 classes -> macro average, AUC is one-vs-rest
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, acc, auc, prec, rec, f1, mcc])
    joblib.dump(model, 'model/' + file_names[name])
    print(name, 'done')

results_df = pd.DataFrame(results, columns=['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
print(results_df.round(4).to_string(index=False))
results_df.round(4).to_csv('model/model_results.csv', index=False)
