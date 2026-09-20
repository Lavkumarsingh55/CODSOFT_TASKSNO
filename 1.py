"""
Task 1: Titanic Survival Prediction
------------------------------------
Builds a machine learning model that predicts whether a passenger
on the Titanic survived, based on features like age, sex, class,
fare, and cabin info.

Usage:
    - If you have Kaggle's train.csv (and optionally test.csv), place
      them in the same folder as this script and run it.
    - If no train.csv is found, the script automatically falls back
      to seaborn's built-in Titanic dataset so it still runs end to end.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
def load_data():
    if os.path.exists("train.csv"):
        print("Loading local train.csv ...")
        df = pd.read_csv("train.csv")
    else:
        print("No train.csv found -> loading seaborn's built-in Titanic dataset instead.")
        import seaborn as sns
        df = sns.load_dataset("titanic")
        # Rename/standardize columns to match the classic Kaggle schema
        df = df.rename(columns={
            "survived": "Survived",
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked",
            "class": "Class",
            "who": "Who",
            "deck": "Cabin",
        })
    return df


# ---------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Family size and "traveling alone" flag
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract title from Name if available (Kaggle dataset only)
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
        rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
        df["Title"] = df["Title"].replace(rare_titles, "Rare")

    # Cabin -> deck letter, and a flag for whether cabin is known
    if "Cabin" in df.columns:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].astype(str).str[0]
        df.loc[df["Deck"].isin(["n", "nan"]), "Deck"] = "Unknown"

    return df


# ---------------------------------------------------------------------
# 3. Build preprocessing + model pipeline
# ---------------------------------------------------------------------
def build_pipeline(numeric_features, categorical_features, model):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------
def main():
    df = load_data()
    df = engineer_features(df)

    target_col = "Survived"
    df = df.dropna(subset=[target_col])  # need a label to train on
    y = df[target_col].astype(int)

    # Pick a reasonable, commonly-available set of features
    candidate_numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin"]
    candidate_categorical = ["Pclass", "Sex", "Embarked", "Title", "Deck"]

    numeric_features = [c for c in candidate_numeric if c in df.columns]
    categorical_features = [c for c in candidate_categorical if c in df.columns]

    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    best_name, best_score, best_pipeline = None, -1, None

    for name, model in models.items():
        pipeline = build_pipeline(numeric_features, categorical_features, model)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(pipeline, X, y, cv=5)

        print(f"\n=== {name} ===")
        print(f"Test accuracy: {acc:.4f}")
        print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("Classification report:")
        print(classification_report(y_test, preds, target_names=["Did not survive", "Survived"]))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, preds))

        if acc > best_score:
            best_name, best_score, best_pipeline = name, acc, pipeline

    print(f"\nBest model: {best_name} (test accuracy = {best_score:.4f})")

    # ------------------------------------------------------------
    # Optional: predict on Kaggle's test.csv if it exists
    # ------------------------------------------------------------
    if os.path.exists("test.csv"):
        print("\nFound test.csv -> generating predictions for submission...")
        test_df = pd.read_csv("test.csv")
        test_df_eng = engineer_features(test_df)

        for col in numeric_features + categorical_features:
            if col not in test_df_eng.columns:
                test_df_eng[col] = np.nan

        X_submit = test_df_eng[numeric_features + categorical_features]
        submit_preds = best_pipeline.predict(X_submit)

        submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": submit_preds,
        })
        submission.to_csv("submission.csv", index=False)
        print("Saved predictions to submission.csv")


if __name__ == "__main__":
    main()