#!/usr/bin/env python
# coding: utf-8

# # Establishing general data boundaries and tendencies

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, label_type=location, rotation=rotation)
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.set_yticklabels(())

def training_classification():
    rfc = RandomForestClassifier()
    abc = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    etc = ExtraTreesClassifier()
    lgr = LogisticRegression()
    svc = SVC()
    mnb = MultinomialNB()
    xgb = XGBClassifier()
    lgb = LGBMClassifier(verbose=-100)
    cat = CatBoostClassifier(verbose=False)

    models = [rfc, abc, gbc, etc, lgr, svc, mnb, xgb, lgb, cat]

    names = ["Random Forest", "Ada Boost", "Gradient Boosting", "Extra Trees", "Logistic Regression",
            "SVC", "Naive Bayes", "XGBoost", "LightGBM", "Cat Boost"]

    scores = []
    cms = dict()
    reports = dict()

    for i, j in enumerate(names):
        models[i].fit(x_train, y_train)
        pred = models[i].predict(x_test)
        scores += [accuracy_score(pred, y_test)]
        cms[j] = confusion_matrix(pred, y_test)
        reports[j] = classification_report(pred, y_test)

    dt = pd.DataFrame({"scores": scores}, index=names)
    dt = dt.sort_values("scores", ascending=False)

    dt["scores"] = dt["scores"]*100
    dt["scores"] = round(dt["scores"], 2)

    fig, axes = plt.subplots(figsize=(15, 6))

    dt["scores"].plot(kind="bar", ax=axes)
    bar_labels(axes)

    index = 0

    for _ in range(2):
        fig, axes = plt.subplots(ncols=5, figsize=(15, 6))
        for i in range(5):
            sns.heatmap(cms[dt.index[index]], annot=True, ax=axes[i])
            axes[i].set_title("{}: {}%".format(dt.index[index], dt.iloc[index, 0]))
            index += 1
        plt.tight_layout()
        plt.show()

    for i in dt.index:
        print("*"*30)
        print("\n")
        print(i)
        print("\n")
        print(reports[i])
    
    return dt  # Return the results DataFrame

# Load data
df = pd.read_csv("data/synthetic_liver_cancer_dataset.csv")

cats = [i for i in df.columns if df[i].nunique() <= 3]
nums = [i for i in df.columns if i not in cats]

index = 0

for _ in range(2):
    fig, axes = plt.subplots(ncols=7, figsize=(15, 6))
    for i in range(7):
        if df.columns[index] in cats:
            df[df.columns[index]].value_counts().plot(kind="bar", ax=axes[i])
            bar_labels(axes[i])
            axes[i].set_title(df.columns[index].replace('_', ' '))
        else:
            sns.histplot(df, x=df.columns[index], kde=True, ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].set_title(df.columns[index].replace('_', ' '))
        index += 1
    plt.tight_layout()
    plt.show()

# # Patients with and without cancer stats

grouped = df.groupby(cats[-1])

index = 0

for j in [4, 3, 3, 3]:
    fig, axes = plt.subplots(ncols=j, figsize=(15, 6))
    for i in range(j):
        if df.columns[index] in cats:
            grouped[df.columns[index]].value_counts().unstack().plot(kind="bar", stacked=True, ax=axes[i])
            bar_labels(axes[i], 0, "center")
            axes[i].set_title(df.columns[index].replace('_', ' '))
        else:
            sns.kdeplot(df, x=df.columns[index], hue=cats[-1], ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].set_title(df.columns[index].replace('_', ' '))
        index += 1
    plt.tight_layout()
    plt.show()

# # Using ML to establish cancer likelihood in patients - different ML models assessment

for i in cats[:-1]:
    df[i] = LabelEncoder().fit_transform(df[i].values)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# Run training and get results
dt = training_classification()

# Save best model and feature names for deployment
import pickle

best_model_name = dt.index[0]
print(f"\n\nBest model is: {best_model_name}")

# Re-train the best model
if best_model_name == "Random Forest":
    best_model = RandomForestClassifier()
elif best_model_name == "Ada Boost":
    best_model = AdaBoostClassifier()
elif best_model_name == "Gradient Boosting":
    best_model = GradientBoostingClassifier()
elif best_model_name == "Extra Trees":
    best_model = ExtraTreesClassifier()
elif best_model_name == "Logistic Regression":
    best_model = LogisticRegression()
elif best_model_name == "SVC":
    best_model = SVC(probability=True)
elif best_model_name == "Naive Bayes":
    best_model = MultinomialNB()
elif best_model_name == "XGBoost":
    best_model = XGBClassifier()
elif best_model_name == "LightGBM":
    best_model = LGBMClassifier(verbose=-100)
else:  # CatBoost
    best_model = CatBoostClassifier(verbose=False)

best_model.fit(x_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save feature names
feature_names = [col for col in df.columns if col != cats[-1]]
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("✅ Model saved as 'model.pkl'")
print(f"✅ Features saved in 'feature_names.pkl': {feature_names}")
