# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:37:43 2021

@author: 82109
"""

def cluster_eval(true_labels, predicted_lable, measure):
        """

        :param true_labels: the real cluster id of data as a list
        :param predicted_lable: predicted cluster id of data as a list
        :param method: the type of evaluation metric
            purity, v-measer, ARI, NMI, AMI
        :return: result value by selected measure
        """
        from sklearn.metrics import v_measure_score
        from sklearn.metrics import adjusted_rand_score
        from sklearn.metrics import normalized_mutual_info_score
        from sklearn.metrics import adjusted_mutual_info_score


        if measure == 'purity':
            purity = purity_score(true_labels, predicted_lable)
            return purity

        elif measure == 'v_meausre':
            v_score = v_measure_score(true_labels, predicted_lable)
            return v_score

        elif measure == 'ARI':
            ARI = adjusted_rand_score(true_labels, predicted_lable)
            return ARI

        elif measure == 'NMI':
            NMI = normalized_mutual_info_score(true_labels, predicted_lable, average_method='arithmetic')
            return NMI

        elif measure == 'AMI':
            AMI = adjusted_mutual_info_score(true_labels, predicted_lable, average_method='arithmetic')
            return AMI

def purity_score(y_true, y_pred):
    """

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: purity score
    """
    import numpy as np
    from sklearn import metrics

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def extracting_predict(all_predicted_label, row_num):
    predicted_label =[]
    for idx in row_num:
        predicted_label.append(all_predicted_label[idx])

    return predicted_label


def calculate_avg_std(evaluation_matrixs, axis=None):
    import numpy as np
    AvgMatrix = np.mean(evaluation_matrixs, axis = 0)
    StdMatrix = np.std(evaluation_matrixs, axis=0)

    return AvgMatrix, StdMatrix

def TimeAvgStd(time_matrix, axis = None):
    import numpy as np

    AvgMatrix = np.mean(time_matrix[1:], axis=0)
    StdMatrix = np.std(time_matrix[1:], axis = 0)

    return AvgMatrix, StdMatrix

def evaluate_labeling_result(split_option, true_label_dict, pred_label):
    '''
    Evaluate labeling result as accuracy and F1 score
    :param split_option:
    :param true_label_dict:
    :param pred_label:
    :return:
    '''
    # data
    true_label = true_label_dict[split_option]

    print("Predicted", pred_label)
    print("True", true_label)

    count = 0
    for i in range(len(true_label)):
        true_list = true_label[i]
        pred_list = pred_label[i]

        for j in range(len(true_list)):
            if pred_list[0] == true_list[j]:
                print("Cluster ",i)
                count+=1

    acc = count / len(true_label) * 100
    f1_score = 0

    return acc, f1_score

def evaluate_labeling_result_past(split_option, true_label_dict, pred_label):
    '''
    Evaluate labeling result as accuracy and F1 score
    :param split_option:
    :param true_label_dict:
    :param pred_label:
    :return:
    '''
    # data
    true_label = true_label_dict[split_option]

    print("Predicted", pred_label)
    print("True", true_label)

    TP = 0
    FP = 0
    FN = 0

    for candidate in pred_label:
        if candidate in true_label:
            TP += 1
        elif candidate not in true_label:
            FP +=1

    for true in true_label:
        if true not in list(pred_label):
            FN += 1

    acc = TP / len(true_label) # (TP+TN) / (TP+TN+TP+FP)
    precision = TP / (TP+FP) # TP / (TP+FP)
    recall = TP / (TP+FN)  # TP / (TP+FN)
    f1_score = 2*precision*recall/(precision+recall)

    return acc, f1_score