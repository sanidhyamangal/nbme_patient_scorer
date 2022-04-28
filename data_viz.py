"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import matplotlib.pyplot as plt  # for plotting
import numpy as np  # numpy analysis
import pandas as pd  # for dataframe loading
import seaborn as sns  # for graph syling


def plot_case_pn_dist(pn_df, case, plot_name: str = "plot_case_pn_dist.png"):

    notes_counts = pn_df.groupby('case_num').count()

    plt.figure(figsize=(20, 8))
    sns.barplot(x=case, y=notes_counts.pn_num, palette='mako')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Case Number', fontsize=15)
    plt.ylabel('Patients Number', fontsize=15)
    plt.title('Distribution of the Patient Notes per Case', fontsize=15)
    plt.savefig(plot_name)
    plt.clf()


def plot_case_feature_dist(features,
                           case,
                           plot_name: str = "plot_case_feature_dist.png"):
    feature_counts = features.groupby('case_num').count()
    plt.figure(figsize=(20, 8))
    sns.barplot(x=case, y=feature_counts.feature_num, palette='mako')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Case Number', fontsize=15)
    plt.ylabel('Features Number', fontsize=15)
    plt.title('Distribution of the Feature Texts per Case', fontsize=15)

    plt.savefig(plot_name)
    plt.clf()


def plot_text_len_dist(pn_note, plot_name: str = "text_len_dist.png"):
    length = []

    for i in range(len(pn_note)):
        length.append(len(pn_note.pn_history[i]))

    plt.figure(figsize=(20, 5))
    plt.title('Distribution of the Feature Text Length', fontsize=15)
    sns.histplot(length)
    plt.xticks(np.arange(0, 1001, 200), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Text Length', fontsize=15)
    plt.ylabel('Count', fontsize=15)

    plt.savefig(plot_name)
    plt.clf()


def plot_feature_dist(features, plot_name="feature_len_dist.png"):
    length = []

    for i in range(len(features)):
        length.append(len(features.feature_text[i]))

    plt.figure(figsize=(20, 5))
    plt.title('Distribution of the Feature Text Length', fontsize=15)
    sns.histplot(length)
    plt.xticks(np.arange(0, 101, 20), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Text Length', fontsize=15)
    plt.ylabel('Count', fontsize=15)

    plt.tight_layout()

    plt.savefig(plot_name)
    plt.clf()


if __name__ == "__main__":
    case = [
        'Case 0', 'Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 6',
        'Case 7', 'Case 8', 'Case 9'
    ]
    pn_note = pd.read_csv("data/patient_notes.csv")
    pn_note['text_len'] = pn_note.pn_history.map(lambda x: len(x))
    features = pd.read_csv("data/features.csv")
    features['text_len'] = features.feature_text.map(lambda x: len(x))

    plot_case_pn_dist(pn_note, case, "case_png_dist.png")
    plot_case_feature_dist(features, case, "case_feature_dist.png")
    plot_feature_dist(features)
    plot_text_len_dist(pn_note)
