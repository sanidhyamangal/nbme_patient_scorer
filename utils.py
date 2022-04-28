"""
author:Sanidhya Mangal
github:sanidhyamangal
"""


def get_model_performance(y_true, y_pred):
    model_performance = y_true - y_pred

    total = model_performance.shape[0]

    neg_count = 0

    for i in model_performance:
        if sum(i) != 0:
            neg_count += 1

    return (1 - (neg_count / total))
