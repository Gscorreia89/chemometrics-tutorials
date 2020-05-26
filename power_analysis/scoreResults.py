import sys
import numpy as np

score_metrics = ['True Positive Rate', 'False Positive Rate', 'True Negative Rate', 'False Negative Rate',
           'Positive Predictive Value', 'Negative Predictive Value', 'False Discovery Rate', 'False Omission Rate',
           'Accuracy', 'F1']


def score_confusionmetrics(result_vector, expected_hits, weight_vector, alpha=0.05):
    """

    Function that calculates performance metrics from an analysis.
    The statistics calculated are True Positive Rate (TPR), True Negative Rate (TNR),
    False Positive Rate (FPR), False Negative Rate (FNR), Positive predictive Value (PPV),
    Negative Predictive Value (NPV), False Discovery Rate (FDR), False Omission rate (FOR), Accuracy and F1 Score.

    Different weighting schemes are possible.
    :param numpy.ndarray result_vector: Vector with the test statistic or p-value for each variable.
    :param numpy.ndarray expected_hits: Vector of 0/1 or Boolean values. 1/True marks which variables are expected
    as hits, 0/False as non-hits.
    :param numpy.ndarray weight_vector: Weight vector (usually pearson correlation) to weight each hit if required
    :param float alpha: Significance threshold or cutoff for the result_vector, to calculate positives and negatives.
    :return: Dictionary with calculated statistics.
    :rtype: dict
    """

    try:

        # Should work for single alpha or a vector of alpha with same size as result_vector (different size)
        # everything which exceeds threshold...
        hits = result_vector < alpha
        # everything else...
        non_hits = result_vector >= alpha

        # true and false positives and negatives
        # calculated from expected hits
        tp = hits & expected_hits
        fp = hits & ~expected_hits
        fn = non_hits & expected_hits
        tn = non_hits & ~expected_hits

        # Weight the results vectors
        if weight_vector is not None:
            tp *= weight_vector
            fp *= weight_vector
            fn *= weight_vector
            tn *= weight_vector

        # Calculate the rates
        # True positive rate and true negative rate
        tpr = tp.sum() / (tp.sum() + fn.sum())
        tnr = tn.sum() / (tn.sum() + fp.sum())
        # false positive and negative rates
        fpr = 1 - tnr
        fnr = 1 - tpr
        # Positive predictive value and Negative predictive value
        ppv = tp.sum() / (tp.sum() + fp.sum())
        npv = tn.sum() / (fn.sum() + tn.sum())
        # False discovery and false omission rate
        fdr = 1 - ppv
        ndr = 1 - npv

        # Accuracy and F1 score
        accuracy = (tp.sum() + tn.sum()) / (fp.sum() + tp.sum() + tn.sum() + fn.sum())
        f1 = 2 * ((ppv*tpr) / (ppv + tpr))

        score_dict = {'True Positive Rate': tpr,
                      'False Positive Rate': fpr,
                      'True Negative Rate': tnr,
                      'False Negative Rate': fnr,
                      'Positive Predictive Value': ppv,
                      'Negative Predictive Value': npv,
                      'False Discovery Rate': fdr,
                      'False Omission Rate': ndr,
                      'Accuracy': accuracy,
                      'F1': f1}

        return score_dict

    except TypeError as terr:
        raise terr
    except ValueError as verr:
        raise verr
    except Exception as exp:
        raise exp


def score_classification_metrics(classification_results, true_label, weight_vector, positive_class=1):
    """

    Function that calculates performance metrics from an analysis.
    The statistics calculated are True Positive Rate (TPR), True Negative Rate (TNR),
    False Positive Rate (FPR), False Negative Rate (FNR), Positive predictive Value (PPV),
    Negative Predictive Value (NPV), False Discovery Rate (FDR), False Omission rate (FOR), Accuracy and F1 Score.

    Different weighting schemes are possible.
    :param numpy.ndarray classification_results: Vector with the classification for each sample.
    :param numpy.ndarray true_label: Vector of 0/1 or Boolean values. 1/True marks which samples are expected
    class hits, 0/False as non-hits.
    :param numpy.ndarray weight_vector: Weight vector (usually pearson correlation) to weight each hit if required
    :return: Dictionary with calculated statistics.
    :rtype: dict
    """

    try:

        # Should work for single alpha or a vector of alpha with same size as result_vector (different size)
        # everything which exceeds threshold...
        # everything else...

        # true and false positives and negatives
        # calculated from expected hits
        tp = (classification_results == true_label) & (classification_results == positive_class)
        fp = (classification_results != true_label) & (classification_results == positive_class)
        fn = (classification_results != true_label) & (classification_results != positive_class)
        tn = (classification_results == true_label) & (classification_results != positive_class)

        # Weight the results vectors
        if weight_vector is not None:
            tp *= weight_vector
            fp *= weight_vector
            fn *= weight_vector
            tn *= weight_vector

        # Calculate the rates
        # True positive rate and true negative rate
        tpr = tp.sum() / (tp.sum() + fn.sum())
        tnr = tn.sum() / (tn.sum() + fp.sum())
        # false positive and negative rates
        fpr = 1 - tnr
        fnr = 1 - tpr
        # Positive predictive value and Negative predictive value
        ppv = tp.sum() / (tp.sum() + fp.sum())
        npv = tn.sum() / (fn.sum() + tn.sum())
        # False discovery and false omission rate
        fdr = 1 - ppv
        ndr = 1 - npv

        # Accuracy and F1 score
        accuracy = (tp.sum() + tn.sum()) / (fp.sum() + tp.sum() + tn.sum() + fn.sum())
        f1 = 2 * ((ppv*tpr) / (ppv + tpr))

        score_dict = {'True Positive Rate': tpr,
                      'False Positive Rate': fpr,
                      'True Negative Rate': tnr,
                      'False Negative Rate': fnr,
                      'Positive Predictive Value': ppv,
                      'Negative Predictive Value': npv,
                      'False Discovery Rate': fdr,
                      'False Omission Rate': ndr,
                      'Accuracy': accuracy,
                      'F1': f1}

        return score_dict

    except TypeError as terr:
        raise terr
    except ValueError as verr:
        raise verr
    except Exception as exp:
        raise exp
