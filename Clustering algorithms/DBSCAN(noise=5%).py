import numpy as np
import time
import math
import random
import copy
from sklearn import metrics
from sklearn.neighbors import KDTree
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import DBSCAN
import pandas as pd
from collections import Counter


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0],ind[1])]) * 1.0 / y_pred.size


def findKNN1(inst, r, tree):
    _, ind = tree.query([inst], r)
    return _[0]


if __name__ == "__main__":
    temp1 = []
    Sil = []
    CH = []
    DB = []
    ARI = []
    AMI = []
    HOM = []
    COM = []
    V = []
    FMI = []
    ACC = []
    NMI = []
    algorithm_consume_time = []
    for xunhuan in range(10):
        txtDF1 = pd.read_csv(r'E:\Datasets\D31.csv')
        col = txtDF1.shape[1]
        df = txtDF1.iloc[:, 0:(col - 1)]
        df_min_max = df.values
        final_true_label1 = txtDF1.iloc[:, (col - 1): col]
        true_value = final_true_label1.values.tolist()
        true_value_1 = []
        for mi in range(len(true_value)):
            true_value_1.append(true_value[mi][0])
        kind1 = []
        for kin in range(txtDF1.shape[0]):
            kind1.append(final_true_label1.values.tolist()[kin][0])
        NC = len(set(kind1))
        minPts = max(1, math.ceil(0.01*len(true_value_1)))

        dist_minPts = []
        n_tree1 = KDTree(df_min_max)
        for ele_point in df_min_max:
            dist_minPts.append(findKNN1(ele_point, minPts+1, n_tree1)[minPts])
        dist_minPts.sort(reverse=True)
        r_value = dist_minPts[math.ceil(len(true_value_1) * 0.05)]
        start_1 = time.perf_counter()
        clf = DBSCAN(eps=r_value, min_samples=minPts)
        clf.fit(df_min_max)
        end_1 = time.perf_counter()
        labels1 = clf.labels_
        labels = []
        for i1 in range(len(labels1)):
            labels.append(int(labels1[i1]))

        time_1 = end_1 - start_1
        del_list = [h_1 for h_1, s_1 in enumerate(labels) if s_1 == -1]
        df_min_max_list = df_min_max.tolist()
        df_label = pd.DataFrame(labels, columns=['label'])
        df_data = pd.DataFrame(df_min_max_list)
        D_data_result = pd.concat([df_data, df_label], axis=1)
        for n_index in reversed(del_list):
            labels.pop(n_index)
            df_min_max_list.pop(n_index)
            true_value_1.pop(n_index)
        if len(set(labels)) == NC:
            algorithm_consume_time.append(time_1)
            Sil.append(metrics.silhouette_score(np.array(df_min_max_list), np.array(labels), metric='euclidean'))
            CH.append(metrics.calinski_harabasz_score(np.array(df_min_max_list), np.array(labels)))
            DB.append(metrics.davies_bouldin_score(np.array(df_min_max_list), np.array(labels)))
            ARI.append(metrics.adjusted_rand_score(np.array(true_value_1), np.array(labels)))
            AMI.append(metrics.adjusted_mutual_info_score(np.array(true_value_1), np.array(labels)))
            HOM.append(metrics.homogeneity_score(np.array(true_value_1), np.array(labels)))
            COM.append(metrics.completeness_score(np.array(true_value_1), np.array(labels)))
            V.append(metrics.v_measure_score(np.array(true_value_1), np.array(labels)))
            FMI.append(metrics.fowlkes_mallows_score(np.array(true_value_1), np.array(labels)))
            ACC.append(acc(np.array(true_value_1), np.array(labels)))
            NMI.append(metrics.normalized_mutual_info_score(np.array(true_value_1), np.array(labels)))
        else:
            break

    temp1.append([np.mean(Sil)])
    temp1.append([np.mean(CH)])
    temp1.append([np.mean(DB)])
    temp1.append([np.mean(ARI)])
    temp1.append([np.mean(AMI)])
    temp1.append([np.mean(HOM)])
    temp1.append([np.mean(COM)])
    temp1.append([np.mean(V)])
    temp1.append([np.mean(FMI)])

    temp1.append([np.mean(ACC)])
    temp1.append([np.mean(NMI)])
    temp1.append([np.mean(algorithm_consume_time)])

    store1 = pd.DataFrame(np.array(temp1).T.tolist(),
    columns = ['Sil', 'CH', 'DB', 'ARI', 'AMI', 'HOM', 'COM', 'V', 'FMI', "ACC", "NMI",  'runtime'])
    store1.to_csv(r'E:\Datasets\DBSCAN-D31(5%).csv',index = False)
