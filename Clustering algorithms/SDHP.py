import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
import matplotlib.ticker as ticker
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment as linear_assignment
from NaN import nan_2
import math


def findKNN1(inst, r, tree):
    _, ind = tree.query([inst], r)
    return _[0]


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
    average_runtime =[]
    for xunhuan in range(10):
        para_list_noise_1 = [0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        para_list_center_1 = [1, 0.95, 0.75, 0.50, 0.25, 0.05, 0.01]

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
        distribution_value = []
        xx1 = []

        zz = 1
        for cpor1 in para_list_center_1:
            x1 = "%.f%%" % (cpor1 * 100)
            z1 = "%.f%%" % (para_list_noise_1[zz-1] * 100)
            txtDF1 = pd.read_csv(r'E:\Datasets\moons(0).csv')
            txtDF1.drop_duplicates(inplace=True)
            col = txtDF1.shape[1]

            final_true_data1 = txtDF1.iloc[:, 0:(col - 1)]
            final_true_label1 = txtDF1.iloc[:, (col - 1): col]
            e_final_true_label1 = []
            for et in range(len(np.array(final_true_label1))):
                e_final_true_label1.append(np.array(final_true_label1)[et][0])

            df_data1 = txtDF1.iloc[:, 0:(col - 1)]
            ar_data1 = np.array(df_data1)
            df_label1 = txtDF1.iloc[:, (col - 1):col]
            ar_label1 = np.array(df_label1)

            nat_neighbor1 = nan_2.Natural_Neighbor()
            nat_neighbor1.load(r'E:\Datasets\moons(0)_1.csv')
            nane1, temp_density1 = nat_neighbor1.algorithm()

            final_density1 = []
            for i in temp_density1:
                final_density1.append(i / nane1)
            df_density1 = pd.DataFrame(final_density1, columns=['Density'])
            D_data_all1 = pd.concat([txtDF1, df_density1], axis=1)

            D_data_all1_sort = D_data_all1.sort_values('Density', ascending=False)
            D_data_all1_sort.to_csv(r'E:\Datasets\sorted_density+T_moons(0).csv',
                index=False)
            D_data_all1_sort1 = pd.read_csv(r'E:\Datasets\sorted_density+T_moons(0).csv')

            kind1 = []
            for kin in range(txtDF1.shape[0]):
                kind1.append(ar_label1.tolist()[kin][0])
            NC1 = len(set(kind1))

            c_count1 = math.ceil(len(D_data_all1_sort1) * cpor1)
            fc_count1 = np.max([c_count1, NC1])
            c_sample1 = D_data_all1_sort1.head(fc_count1)
            df_center1 = c_sample1.iloc[:, 0:(D_data_all1_sort1.shape[1]-2)]
            df_carr1 = df_center1.values
            df_cli1 = df_carr1.tolist()

            remain_sample_count1 = len(D_data_all1_sort1) - fc_count1
            r_sample1 = D_data_all1_sort1.loc[fc_count1:(len(D_data_all1_sort1) - 1)]
            df_noise1 = r_sample1.iloc[:, 0:(D_data_all1_sort1.shape[1]-2)].values.tolist()

            start_1 = time.perf_counter()
            estimator1 = AgglomerativeClustering(n_clusters=NC1, linkage='single')
            estimator1.fit(df_carr1)

            Center_Clusters1 = [[] for i in range(NC1)]
            for c_nc1 in range(NC1):
                for cpoint1, cid1 in zip(df_cli1, estimator1.labels_):
                    if c_nc1 == cid1:
                        Center_Clusters1[c_nc1].append(cpoint1)
                    else:
                        continue

            if len(df_noise1) > 0:
                noise_class1 = []
                for noise_point1 in df_noise1:
                    dist_naverage1 = []
                    for n_cluster1 in Center_Clusters1:
                        n_tree1 = KDTree(np.array(n_cluster1))
                        min_nnane1 = np.min([nane1, len(n_cluster1)])
                        dist_naverage1.append(np.mean(findKNN1(np.array(noise_point1), min_nnane1, n_tree1)))
                    temp_nid1 = dist_naverage1.index(min(dist_naverage1))
                    noise_class1.append(temp_nid1)
                    Center_Clusters1[temp_nid1].append(noise_point1)
            end_1 = time.perf_counter()

            for r1 in range(NC1):
                for s1 in range(len(Center_Clusters1[r1])):
                    Center_Clusters1[r1][s1].append(r1)
            distribution_1 = []
            for dis_1 in Center_Clusters1:
                distribution_1.append(len(dis_1))
            dis_min = np.min(distribution_1)
            dis_max = np.max(distribution_1)
            distribution_value.append(dis_max-dis_min)

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
            zz = zz + 1
            final_eve_data = []
            for pre_point1, pre_cid1 in zip(f_ar_data1.tolist(), np.array(predict_label_1)):
                for t_point1, t_cid1 in zip(final_true_data1.values.tolist(), np.array(e_final_true_label1)):
                    if pre_point1 == t_point1:
                        pre_point1.append(pre_cid1)
                        pre_point1.append(t_cid1)
                    else:
                        continue
                final_eve_data.append(pre_point1)
            ex_all_data1 = pd.DataFrame(final_eve_data)
            ex_all_data1.to_csv(r'E:\Datasets\moons(0)PT.csv', index=False)

            pre_label1 = ex_all_data1.iloc[:, (len(final_eve_data[0])-2):(len(final_eve_data[0])-1)]
            ture_label1 = ex_all_data1.iloc[:, (len(final_eve_data[0])-1):len(final_eve_data[0])]
            eva_data1 = ex_all_data1.iloc[:, 0:(len(final_eve_data[0])-2)]
            f_predict_label_1 =[]
            f_ture_label_1 = []
            for oe in range(len(pre_label1)):
                f_predict_label_1.append(int(pre_label1.values[oe][0]))
                f_ture_label_1.append(int(ture_label1.values[oe][0]))

            xx1.append(c_count1)
            ACC1.append(acc(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            NMI1.append(metrics.normalized_mutual_info_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            ARI1.append(metrics.adjusted_rand_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            AMI1.append(metrics.adjusted_mutual_info_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            Sil1.append(metrics.silhouette_score(np.array(eva_data1), np.array(f_predict_label_1), metric='euclidean'))
            CH1.append(metrics.calinski_harabasz_score(np.array(eva_data1), np.array(f_predict_label_1)))
            HOM1.append(metrics.homogeneity_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            COM1.append(metrics.completeness_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            V1.append(metrics.v_measure_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            FMI1.append(metrics.fowlkes_mallows_score(np.array(f_ture_label_1), np.array(f_predict_label_1)))
            DB1.append(metrics.davies_bouldin_score(np.array(eva_data1), np.array(f_predict_label_1)))
            runtime1.append(end_1 - start_1)

            if fc_count1 == NC1:
                break
            else:
                continue

        temp1 = []
        temp1.append(xx1)
        temp1.append(distribution_value)

        temp1.append(Sil1)
        temp1.append(CH1)
        temp1.append(DB1)

        temp1.append(ARI1)
        temp1.append(AMI1)
        temp1.append(HOM1)
        temp1.append(COM1)
        temp1.append(V1)
        temp1.append(FMI1)

        temp1.append(ACC1)
        temp1.append(NMI1)

        temp1.append(runtime1)
        average_runtime.append(np.sum(runtime1))

        store1 = pd.DataFrame(np.array(temp1).T.tolist(),
                              columns=['number of core points', "Distribution difference", 'Sil', 'CH', 'DB','ARI', 'AMI',
                                       'HOM', 'COM','V','FMI', "ACC", "NMI", 'runtime'])
        store1.to_csv(r'E:\Datasets\moons(0)PV.csv', index=False)
    print("Runtime：", np.mean(average_runtime))