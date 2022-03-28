"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import pandas as pd # for csv reading


def load_train_dataset():
    patient_notes = pd.read_csv("data/patient_notes.csv")
    train_data = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")

    train_dataset = train_data.merge(features, on=["feature_num", "case_num"], how="left")
    train_dataset = train_dataset.merge(patient_notes, how="left")

    train_dict = []

    for i in train_dataset.groupby("pn_num"):
        train_dict.append({
            "pn_num":i[1]["pn_num"].values[0],
            "feature_num":i[1]["feature_num"].values,
            "pn_history":i[1]["pn_history"].values[0]
        })

    return pd.DataFrame(train_dict)