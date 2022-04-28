"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import pandas as pd  # for csv reading
import numpy as np  # for matrix maths


def process_features(feature_nums):
    features = feature_nums.values

    return np.pad(features, (0, (18 - len(features))))


def load_train_dataset():
    patient_notes = pd.read_csv("data/patient_notes.csv")
    train_data = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")

    train_dataset = train_data.merge(features,
                                     on=["feature_num", "case_num"],
                                     how="left")
    train_dataset = train_dataset.merge(patient_notes, how="left")

    train_dict = []

    for i in train_dataset.groupby("pn_num"):
        train_dict.append({
            "pn_num": i[1]["pn_num"].values[0],
            "feature_num": process_features(i[1]["feature_num"]),
            "pn_history": i[1]["pn_history"].values[0],
            "pn_case": i[1]["case_num"].values[0]
        })

    return pd.DataFrame(train_dict)
