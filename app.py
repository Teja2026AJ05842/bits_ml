import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="Frog Call Species Classifier", layout="wide")

TARGET = "Species"

saved_classifiers = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}

# the 10 species in the dataset (label as in the csv : family / genus and a short description)
species_info = {
    "AdenomeraHylaedactylus": "Leptodactylidae / Adenomera - most common species in the data (3478 calls)",
    "HypsiboasCordobae": "Hylidae / Hypsiboas - tree frog, 2nd largest class",
    "AdenomeraAndre": "Leptodactylidae / Adenomera - same genus as the biggest class, so easy to confuse",
    "Ameeregatrivittata": "Dendrobatidae / Ameerega - three-striped poison frog",
    "HypsiboasCinerascens": "Hylidae / Hypsiboas - tree frog",
    "HylaMinuta": "Hylidae / Dendropsophus - lesser tree frog",
    "LeptodactylusFuscus": "Leptodactylidae / Leptodactylus - whistling frog",
    "ScinaxRuber": "Hylidae / Scinax - red snouted tree frog",
    "OsteocephalusOophagus": "Hylidae / Osteocephalus - tree frog, only 114 calls",
    "Rhinellagranulosa": "Bufonidae / Rhinella - granular toad, rarest class (68 calls)",
}


@st.cache_resource
def read_pickle(path):
    return joblib.load(path)


def calc_scores(y_true, pred, prob, classes):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=classes),
        "Precision": precision_score(y_true, pred, average="macro"),
        "Recall": recall_score(y_true, pred, average="macro"),
        "F1": f1_score(y_true, pred, average="macro"),
        "MCC": matthews_corrcoef(y_true, pred),
    }


# family of each species, taken from the text above (part before " / ")
species_family = {k: v.split(" / ")[0] for k, v in species_info.items()}

# ---------- header ----------
st.title("Frog Call Species Classifier")
st.write("2026AJ05842 | Sai Teja Koneru | ML Assignment 2 | Dataset: Kaggle Anuran Calls (MFCCs) - "
         "7195 frog call syllables, 22 MFCC audio features, 10 species")

# ---------- inputs (top of the page instead of sidebar) ----------
in1, in2 = st.columns([2, 1])
with in1:
    csv_file = st.file_uploader("Upload test data (CSV)", type="csv")
with in2:
    st.write("")
    st.write("")
    # button only fires once per click, so remember the choice in session_state
    if st.button("Autofill Test Data"):
        st.session_state["use_default"] = True
    use_default = st.session_state.get("use_default", False)
st.caption("The test CSV must have the 22 MFCC columns (MFCC_1 ... MFCC_22) and the Species column.")

with st.expander("Species / class labels"):
    for k, v in species_info.items():
        st.write(f"**{k}** - {v}")

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.caption("Using uploaded file: " + csv_file.name)
elif use_default:
    df = pd.read_csv("test_data.csv")
    st.caption("Using test_data.csv from the repo (1799 rows)")
else:
    st.warning("Upload the test CSV above, or click 'Autofill Test Data' to use the test_data.csv from the repo.")
    st.stop()

if TARGET not in df.columns:
    st.error(f"{TARGET} column not found in the csv")
    st.stop()

# in case someone uploads the raw file with the extra label columns
drop_cols = [c for c in ["Family", "Genus", "RecordID"] if c in df.columns]
X = df.drop(columns=[TARGET] + drop_cols)
y = df[TARGET]

# uploaded data overview
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Uploaded data")
    st.write(f"{df.shape[0]} rows, {X.shape[1]} features")
    st.dataframe(df.head(), height=210)
with c2:
    st.subheader("Species counts")
    st.bar_chart(y.value_counts())

st.divider()

# model dropdown, placed right above the results it controls
sel_col, _ = st.columns([1, 2])
with sel_col:
    chosen_clf = st.selectbox("Choose a classifier", list(saved_classifiers.keys()))

# load saved scaler + model (no training here)
scaler = read_pickle("model/scaler.pkl")
model = read_pickle(saved_classifiers[chosen_clf])

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)
classes = model.classes_

st.header(f"Results: {chosen_clf}")

# c. metrics
scores = calc_scores(y, y_pred, y_prob, classes)
cols = st.columns(6)
for col, (metric, value) in zip(cols, scores.items()):
    col.metric(metric, f"{value:.4f}")
st.caption("Precision, Recall and F1 are macro averaged over the 10 species. AUC is one-vs-rest (macro).")

tab1, tab2, tab3, tab5, tab4 = st.tabs(["Confusion Matrix", "Classification Report",
                                        "Per-species Recall", "Family Check", "Predictions"])

# d. confusion matrix and classification report
with tab1:
    cm = confusion_matrix(y, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#eef6ec")  # same as page background
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

with tab2:
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))

with tab3:
    # how many calls of each species were identified correctly
    recall_per_class = pd.Series(recall_score(y, y_pred, average=None, labels=classes), index=classes)
    st.bar_chart(recall_per_class.sort_values())
    worst = recall_per_class.idxmin()
    best = recall_per_class.idxmax()
    st.write(f"Easiest species for {chosen_clf}: **{best}** ({recall_per_class[best]:.0%} recall). "
             f"Hardest: **{worst}** ({recall_per_class[worst]:.0%} recall).")

with tab5:
    # each species belongs to one of 4 families. check if the family is still right when the species is wrong
    true_fam = y.map(species_family)
    pred_fam = pd.Series(y_pred, index=y.index).map(species_family)
    wrong = y != y_pred
    f1c, f2c, f3c = st.columns(3)
    f1c.metric("Species accuracy", f"{(~wrong).mean():.2%}")
    f2c.metric("Family accuracy", f"{(true_fam == pred_fam).mean():.2%}")
    if wrong.sum() > 0:
        same_fam = (true_fam[wrong] == pred_fam[wrong]).mean()
        f3c.metric("Wrong species but right family", f"{same_fam:.0%}")
        if same_fam >= 0.5:
            note = f"so {chosen_clf} mostly mixes up related species."
        else:
            note = f"so {chosen_clf} often confuses species from different families."
        st.write(f"Out of {wrong.sum()} wrong predictions, {(true_fam[wrong] == pred_fam[wrong]).sum()} "
                 f"are still in the correct family, {note}")
    else:
        f3c.metric("Wrong species but right family", "-")
    st.write("Family level confusion (rows = actual family, columns = predicted family):")
    st.dataframe(pd.crosstab(true_fam, pred_fam, rownames=["Actual"], colnames=["Predicted"]))

with tab4:
    pred_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
    pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]
    show_wrong = st.checkbox("Show only wrong predictions")
    if show_wrong:
        st.dataframe(pred_df[~pred_df["Correct"]])
    else:
        st.dataframe(pred_df)
    st.write(f"Wrong predictions: {(~pred_df['Correct']).sum()} out of {len(pred_df)}")
    st.download_button("Download predictions", pred_df.to_csv(index=False),
                       file_name=f"predictions_{chosen_clf.replace(' ', '_').lower()}.csv")

# ---------- call fingerprint: average MFCC values per species ----------
st.divider()
st.header("Call fingerprint (average MFCCs per species)")
st.write("Each line is the average of the 22 MFCC values for one species. Species whose lines are close "
         "together sound similar, so they are harder to tell apart.")
avg_mfcc = X.groupby(y).mean().T
avg_mfcc.index = range(1, len(avg_mfcc) + 1)   # 1..22 so the x axis stays in order
avg_mfcc.index.name = "MFCC number"
default_pick = [s for s in ["AdenomeraAndre", "AdenomeraHylaedactylus", "HylaMinuta"] if s in avg_mfcc.columns]
pick = st.multiselect("Species to compare", list(avg_mfcc.columns), default=default_pick)
if pick:
    st.line_chart(avg_mfcc[pick])
else:
    st.info("Select at least one species to see its fingerprint.")

# ---------- check one call with all 5 models ----------
st.divider()
st.header("Check one call")
if "call_idx" not in st.session_state or st.session_state["call_idx"] >= len(df):
    st.session_state["call_idx"] = 0
if st.button("Pick a random call"):
    st.session_state["call_idx"] = int(pd.Series(range(len(df))).sample(1).iloc[0])
idx = st.session_state["call_idx"]
true_species = y.iloc[idx]
st.write(f"Row {idx} of the test file | actual species: **{true_species}** ({species_family.get(true_species, '?')})")
one_call = []
for name, path in saved_classifiers.items():
    m = read_pickle(path)
    guess = m.predict(X_scaled[idx:idx + 1])[0]
    conf = m.predict_proba(X_scaled[idx:idx + 1]).max()
    one_call.append({"Model": name, "Predicted species": guess, "Confidence": f"{conf:.0%}",
                     "Correct": "✔" if guess == true_species else "✘"})
st.dataframe(pd.DataFrame(one_call), hide_index=True)

# all 5 models on the uploaded data
st.divider()
st.header("All 5 models on this file")
rows = []
for name, path in saved_classifiers.items():
    m = read_pickle(path)
    pred = m.predict(X_scaled)
    prob = m.predict_proba(X_scaled)
    rows.append({"ML Model Name": name, **calc_scores(y, pred, prob, m.classes_)})
all_scores = pd.DataFrame(rows).set_index("ML Model Name").round(4)
# dark green + white text so it is readable in both light and dark theme
st.dataframe(all_scores.style.highlight_max(axis=0, props="background-color: #2e7d32; color: white; font-weight: bold")
             .format("{:.4f}"))
top = all_scores["F1"].idxmax()
st.write(f"Best model by F1 score on this file: {top} (F1 = {all_scores.loc[top, 'F1']:.4f})")
