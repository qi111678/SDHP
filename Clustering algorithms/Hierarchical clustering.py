import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
import matplotlib.ticker as ticker
from sklearn import metrics
from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment as linear_assignment
from NaN import nan_2
import math


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0],ind[1])]) * 1.0 / y_pred.size


if __name__ == "__main__":
    runtime1 = []
    ACC1 = []
    NMI1 = []
    ARI1 = []
    AMI1 = []
    HOM1 = []
    COM1 = []
    V1 = []
    FMI1 = []
    Sil1 = []
    CH1 = []
    DB1 = []
    temp1 = []
    for xunhuan in range(10):
        txtDF1 = pd.read_csv(r'E:\Datasets\D31.csv')
        col = txtDF1.shape[1]
        final_true_data1 = txtDF1.iloc[:, 0:(col-1)]
        final_true_label1 = txtDF1.iloc[:, (col-1): col]
        e_final_true_label1 = []
        for et in range(len(np.array(final_true_label1))):
            e_final_true_label1.append(np.array(final_true_label1)[et][0])

        df_data1 = txtDF1.iloc[:, 0:(col-1)]
        df_carr1 = np.array(df_data1)
        df_cli1 = df_carr1.tolist()
        df_label1 = txtDF1.iloc[:, (col-1):col]
        ar_label1 = np.array(df_label1)

        kind1 = []
        for kin in range(txtDF1.shape[0]):
            kind1.append(ar_label1.tolist()[kin][0])
        NC1 = len(set(kind1))

        start_1 = time.perf_counter()
        estimator1 = AgglomerativeClustering(n_clusters=NC1, linkage='single')
        estimator1.fit(df_carr1)
        end_1 = time.perf_counter()

        Center_Clusters1 = [[] for i in range(NC1)]
        for c_nc1 in range(NC1):
            for cpoint1, cid1 in zip(df_cli1, estimator1.labels_):
                if c_nc1 == cid1:
                    Center_Clusters1[c_nc1].append(cpoint1)
                else:
                    continue

        for r1 in range(NC1):
            for s1 in range(len(Center_Clusters1[r1])):
                Center_Clusters1[r1][s1].append(r1)

        distribution_1 = []
        for dis_1 in Center_Clusters1:
            distribution_1.append(len(dis_1))

        final_clusters = []
        for f1 in Center_Clusters1:
            for ef in f1:
                final_clusters.append(ef)
        df_data_1 = pd.DataFrame(np.array(final_clusters))
        f_df_label1 = df_data_1.iloc[:, (df_data_1.shape[1]-1):df_data_1.shape[1]]
        f_ar_label1 = np.array(f_df_label1)
        predict_label_1 = []
        for oe in range(len(f_ar_label1)):
            predict_label_1.append(int(f_ar_label1[oe][0]))
        f_df_data_11 = df_data_1.iloc[:, 0:(df_data_1.shape[1]-1)]
        f_ar_data1 = np.array(f_df_data_11)

        final_eve_data = []
        for pre_point1, pre_cid1 in zip(f_ar_data1.tolist(), np.array(predict_label_1)):
            for t_point1, t_cid1 in zip(final_true_data1.values.tolist(), np.array(e_final_true_label1)):
                if pre_point1 == t_point1:
                    pre_point1.append(pre_cid1)
                    pre_point1.append(t_cid1)
                else:
                    continue
            final_eve_data.append(pre_point1)
        ex_all_data1 = pd.DataFrame(np.array(final_eve_data))

        pre_label1 = ex_all_data1.iloc[:, (len(final_eve_data[0])-2):(len(final_eve_data[0])-1)]

        ture_label1 = ex_all_data1.iloc[:, (len(final_eve_data[0])-1):len(final_eve_data[0])]
        eva_data1 = ex_all_data1.iloc[:, 0:(len(final_eve_data[0])-2)]
        f_predict_label_1 =[]
        f_ture_label_1 = []
        for oe in range(len(pre_label1)):
            f_predict_label_1.append(int(pre_label1.values[oe][0]))
            f_ture_label_1.append(int(ture_label1.values[oe][0]))

        ACC1.append(acc(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        NMI1.append(metrics.normalized_mutual_info_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        ARI1.append(metrics.adjusted_rand_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        AMI1.append(metrics.adjusted_mutual_info_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        HOM1.append(metrics.homogeneity_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        COM1.append(metrics.completeness_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        V1.append(metrics.v_measure_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        FMI1.append(metrics.fowlkes_mallows_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
        Sil1.append(metrics.silhouette_score(np.array(eva_data1), np.array(f_predict_label_1), metric='euclidean'))
        CH1.append(metrics.calinski_harabasz_score(np.array(eva_data1), np.array(f_predict_label_1)))
        DB1.append(metrics.davies_bouldin_score(np.array(eva_data1), np.array(f_predict_label_1)))
        runtime1.append(end_1 - start_1)

    temp1.append([np.mean(Sil1)])
    temp1.append([np.mean(CH1)])
    temp1.append([np.mean(DB1)])
    temp1.append([np.mean(ARI1)])
    temp1.append([np.mean(AMI1)])
    temp1.append([np.mean(HOM1)])
    temp1.append([np.mean(COM1)])
    temp1.append([np.mean(V1)])
    temp1.append([np.mean(FMI1)])
    temp1.append([np.mean(ACC1)])
    temp1.append([np.mean(NMI1)])
    temp1.append([np.mean(runtime1)])

    store1 = pd.DataFrame(np.array(temp1).T.tolist(),
                          columns=['Sil', 'CH', 'DB', 'ARI', 'AMI', 'HOM', 'COM', 'V', 'FMI', "ACC", "NMI",  'runtime'])
    store1.to_csv(r'E:\Datasets\single-D31.csv',index=False)
