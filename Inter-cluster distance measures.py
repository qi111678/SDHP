import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
import copy
import matplotlib.ticker as ticker
from sklearn import metrics
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
    plt.figure(figsize=(48, 40))
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 57}
    font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': 60}
    hh = 56
    dd = 400
    index=1
    flag=1
    yflag=1
    ccc=1
    zz = 1
    for jj in range(0, 5):
        txtDF0 = pd.read_csv(r'E:\Datasets\Gaussian_4_original{s1}.csv'.format(s1=jj))
        txtDF0.drop_duplicates(inplace=True)
        txtDF0.to_csv(r'E:\Datasets\Gaussian_4_original{s1}.csv'.format(s1=jj),index=False)
        txtDF1 = pd.read_csv(r'E:\Datasets\Gaussian_4_original{s1}.csv'.format(s1=jj))
        col = txtDF1.shape[1]

        final_true_data1 = txtDF1.iloc[:, 0:col]
        df_data1 = txtDF1.iloc[:, 0:col]
        ar_data1 = np.array(df_data1)

        nat_neighbor1 = nan_2.Natural_Neighbor()
        nat_neighbor1.load(r'E:\Datasets\Gaussian_4_original{s1}_1.csv'.format(s1=jj))                                                                                                  # 改2
        nane1, temp_density1 = nat_neighbor1.algorithm()

        final_density1 = []
        for i in temp_density1:
            final_density1.append(i / nane1)

        df_density1 = pd.DataFrame(final_density1, columns=['Density'])
        D_data_all1 = pd.concat([txtDF1, df_density1], axis=1)

        D_data_all1_sort = D_data_all1.sort_values('Density', ascending=False)
        D_data_all1_sort.to_csv(r'E:\Datasets\density+T_Gaussian_4_original{s1}_sorted.csv'.format(s1=jj),index=False)
        para_list_noise_1 = [0.95]

        para_list_center_1 = [0.05]
        D_data_all1_sort1 = pd.read_csv(r'E:\Datasets\density+T_Gaussian_4_original{s1}_sorted.csv'.format(s1=jj))
        NC1 = 4

        for cpor1 in para_list_center_1:
            x1 = "%.f%%" % (cpor1 * 100)
            c_count1 = math.ceil(len(D_data_all1_sort1) * cpor1)
            fc_count1 = np.max([c_count1, NC1])
            c_sample1 = D_data_all1_sort1.head(fc_count1)
            df_center1 = c_sample1.iloc[:, 0:(D_data_all1_sort1.shape[1]-1)]
            df_carr1 = df_center1.values
            df_cli1 = df_carr1.tolist()

            remain_sample_count1 = len(D_data_all1_sort1) - fc_count1
            r_sample1 = D_data_all1_sort1.loc[fc_count1:(len(D_data_all1_sort1) - 1)]
            df_noise1 = r_sample1.iloc[:, 0:(D_data_all1_sort1.shape[1]-1)].values.tolist()

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
                Singel_centers = copy.deepcopy(Center_Clusters1)
                Singel_noise = copy.deepcopy(df_noise1)
                start_1 = time.perf_counter()
                for single_noise_point in Singel_noise:
                    single_distance = []
                    for already_clu in Singel_centers:
                        single_tree = KDTree(np.array(already_clu))
                        single_distance.append(findKNN1(np.array(single_noise_point), 1, single_tree)[0])
                    Singel_centers[single_distance.index(min(single_distance))].append(single_noise_point)

                end_1 = time.perf_counter()
                print("Runtime：",end_1 - start_1)
                final_single_results=[]
                for label_sin in range(4):
                    for ele_sin in Singel_centers[label_sin]:
                        ele_sin.append(label_sin)
                        final_single_results.append(ele_sin)
                Complete_centers = copy.deepcopy(Center_Clusters1)
                Complete_noise = copy.deepcopy(df_noise1)

                start_2 = time.perf_counter()
                for complete_noise_point in Complete_noise:
                    complete_distance = []
                    for already_clu_c in Complete_centers:
                        complete_tree = KDTree(np.array(already_clu_c))
                        complete_distance.append(findKNN1(np.array(complete_noise_point), len(already_clu_c), complete_tree)[len(already_clu_c)-1])
                    Complete_centers[complete_distance.index(min(complete_distance))].append(complete_noise_point)
                end_2 = time.perf_counter()
                print("Runtime：", end_2 - start_2)
                final_complete_results = []
                for label_com in range(4):
                    for ele_com in Complete_centers[label_com]:
                        ele_com.append(label_com)
                        final_complete_results.append(ele_com)
                Average_centers = copy.deepcopy(Center_Clusters1)
                Average_noise = copy.deepcopy(df_noise1)
                start_3 = time.perf_counter()
                for Average_noise_point in Average_noise:
                    Average_distance = []
                    for already_clu_a in Average_centers:
                        Average_tree = KDTree(np.array(already_clu_a))
                        Average_distance.append(np.mean(findKNN1(np.array(Average_noise_point), len(already_clu_a), Average_tree)))
                    Average_centers[Average_distance.index(min(Average_distance))].append(Average_noise_point)
                end_3 = time.perf_counter()
                print("Runtime：", end_3 - start_3)
                final_average_results = []
                for label_avg in range(4):
                    for ele_avg in Average_centers[label_avg]:
                        ele_avg.append(label_avg)
                        final_average_results.append(ele_avg)
                Centroid_centers = copy.deepcopy(Center_Clusters1)
                Centroid_noise = copy.deepcopy(df_noise1)

                start_4 = time.perf_counter()
                for Centroid_noise_point in Centroid_noise:
                    Centroid_distance = []
                    for already_clu_Cen in Centroid_centers:
                        centro_sum = np.array([0, 0])
                        for centro in range(len(already_clu_Cen)):
                            centro_sum = centro_sum + np.array(already_clu_Cen[centro])
                        cluster_cen=centro_sum / len(already_clu_Cen)
                        Centroid_distance.append(np.linalg.norm(cluster_cen-np.array(Centroid_noise_point)))
                    Centroid_centers[Centroid_distance.index(min(Centroid_distance))].append(Centroid_noise_point)
                end_4 = time.perf_counter()
                print("Runtime：", end_4 - start_4)
                final_Centroid_results = []
                for label_Centroid in range(4):
                    for ele_Centroid in Centroid_centers[label_Centroid]:
                        ele_Centroid.append(label_Centroid)
                        final_Centroid_results.append(ele_Centroid)
                # -------------------------------------Ward linkage------------------------
                Ward_centers = copy.deepcopy(Center_Clusters1)
                Ward_noise = copy.deepcopy(df_noise1)
                start_5 = time.perf_counter()
                for Ward_noise_point in Ward_noise:
                    Ward_distance = []
                    for already_clu_war in Ward_centers:
                        war_sum = np.array([0, 0])
                        for wartro in range(len(already_clu_war)):
                            war_sum= war_sum + np.array(already_clu_war[wartro])
                        cluster_war = war_sum / len(already_clu_war)
                        Ward_distance.append((np.linalg.norm(cluster_war - np.array(Ward_noise_point))*len(already_clu_war))/(len(already_clu_war)+1))
                    Ward_centers[Ward_distance.index(min(Ward_distance))].append(Ward_noise_point)
                final_Ward_results = []
                end_5 = time.perf_counter()
                print("Runtime：", end_5 - start_5)
                for label_Ward in range(4):
                    for ele_Ward in Ward_centers[label_Ward]:
                        ele_Ward.append(label_Ward)
                        final_Ward_results.append(ele_Ward)

                noise_class1 = []
                NN_centers = copy.deepcopy(Center_Clusters1)
                NN_noise = copy.deepcopy(df_noise1)
                start_6 = time.perf_counter()
                for noise_point1 in NN_noise:
                    dist_naverage1 = []
                    for n_cluster1 in NN_centers:
                        n_tree1 = KDTree(np.array(n_cluster1))
                        min_nnane1 = np.min([nane1, len(n_cluster1)])
                        dist_naverage1.append(np.mean(findKNN1(np.array(noise_point1), min_nnane1, n_tree1)))
                    temp_nid1 = dist_naverage1.index(min(dist_naverage1))
                    noise_class1.append(temp_nid1)
                    NN_centers[temp_nid1].append(noise_point1)
                end_6 = time.perf_counter()
                print("Runtime：", end_6 - start_6)
                final_NN_results = []
                for label_NN in range(4):
                    for ele_NN in NN_centers[label_NN]:
                        ele_NN.append(label_NN)
                        final_NN_results.append(ele_NN)
            # ----------------------------------------------------------------------------------------------
            Single_results = pd.DataFrame(np.array(final_single_results))
            Complete_results = pd.DataFrame(np.array(final_complete_results))
            Average_results = pd.DataFrame(np.array(final_average_results))
            Centroid_results = pd.DataFrame(np.array(final_Centroid_results))
            Ward_results = pd.DataFrame(np.array(final_Ward_results))
            NN_results = pd.DataFrame(np.array(final_NN_results))
            Single_data = np.array(Single_results.iloc[:, 0:2]).tolist()
            Single_label_1 = np.array(Single_results.iloc[:, 2:3])
            Single_label=[]
            for single_1 in range(len(Single_data)):
                Single_label.append(int(Single_label_1[single_1][0]))

            Complete_data = np.array(Complete_results.iloc[:, 0:2]).tolist()
            Complete_label_1 = np.array(Complete_results.iloc[:, 2:3])
            Complete_label = []
            for Complete_1 in range(len(Complete_data)):
                Complete_label.append(int(Complete_label_1[Complete_1][0]))

            Average_data = np.array(Average_results.iloc[:, 0:2]).tolist()
            Average_label_1 = np.array(Average_results.iloc[:, 2:3])
            Average_label = []
            for Average_1 in range(len(Average_data)):
                Average_label.append(int(Average_label_1[Average_1][0]))

            Centroid_data = np.array(Centroid_results.iloc[:, 0:2]).tolist()
            Centroid_label_1 = np.array(Centroid_results.iloc[:, 2:3])
            Centroid_label = []
            for Centroid_1 in range(len(Centroid_data)):
                Centroid_label.append(int(Centroid_label_1[Centroid_1][0]))

            Ward_data = np.array(Ward_results.iloc[:, 0:2]).tolist()
            Ward_label_1 = np.array(Ward_results.iloc[:, 2:3])
            Ward_label = []
            for Ward_1 in range(len(Ward_data)):
                Ward_label.append(int(Ward_label_1[Ward_1][0]))

            NN_data = np.array(NN_results.iloc[:, 0:2]).tolist()
            NN_label_1 = np.array(NN_results.iloc[:, 2:3])
            NN_label = []
            for NN_1 in range(len(NN_data)):
                NN_label.append(int(NN_label_1[NN_1][0]))


            plt.subplot(5, 6, index)
            center_cnames1 = ['blue', 'red', 'green', "purple"]
            for Single_point1, Single_id1 in zip(Single_data, np.array(Single_label)):
                plt.scatter(Single_point1[0], Single_point1[1], c=center_cnames1[Single_id1], s=dd, alpha=1)

            if flag == 1:
                plt.title("NNLD-SL-SL", fontdict=font, bbox=dict(ec='pink', fc='orange'), verticalalignment='bottom')
                flag=flag+1
            if yflag == 1:
                plt.ylabel('Illustrative dataset 1', font1, bbox=dict(ec='orange', fc='w'), verticalalignment='bottom')
                yflag = 0
            if yflag == 2:
                plt.ylabel('Illustrative dataset 2', font1, bbox=dict(ec='orange', fc='w'), verticalalignment='bottom')
                yflag = 0
            if yflag == 3:
                plt.ylabel('Illustrative dataset 3', font1, bbox=dict(ec='orange', fc='w'), verticalalignment='bottom')
                yflag = 0
            if yflag == 4:
                plt.ylabel('Illustrative dataset 4', font1, bbox=dict(ec='orange', fc='w'), verticalalignment='bottom')
                yflag = 0
            if yflag == 5:
                plt.ylabel('Illustrative dataset 5', font1, bbox=dict(ec='orange', fc='w'), verticalalignment='bottom')
                yflag = 0
            plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.tight_layout()

            plt.subplot(5, 6, index+1)
            center_cnames1 = ['blue', 'red', 'green', "purple"]
            for Complete_point1, Complete_id1 in zip(Complete_data, np.array(Complete_label)):
                plt.scatter(Complete_point1[0], Complete_point1[1], c=center_cnames1[Complete_id1], s=dd, alpha=1)
            if flag == 2:
                plt.title("NNLD-SL-CL", fontdict=font, bbox=dict(ec='pink', fc='orange'), verticalalignment='bottom')
                flag = flag + 1
            plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.tight_layout()

            plt.subplot(5, 6, index + 2)
            center_cnames1 = ['blue', 'red', 'green', "purple"]
            for Average_point1, Average_id1 in zip(Average_data, np.array(Average_label)):
                plt.scatter(Average_point1[0], Average_point1[1], c=center_cnames1[Average_id1], s=dd, alpha=1)
            if flag == 3:
                plt.title("NNLD-SL-AL", fontdict=font, bbox=dict(ec='pink', fc='orange'), verticalalignment='bottom')
                flag = flag + 1
            plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.tight_layout()

            plt.subplot(5, 6, index + 3)
            center_cnames1 = ['blue', 'red', 'green', "purple"]
            for Centroid_point1, Centroid_id1 in zip(Centroid_data, np.array(Centroid_label)):
                plt.scatter(Centroid_point1[0], Centroid_point1[1], c=center_cnames1[Centroid_id1], s=dd, alpha=1)
            if flag == 4:
                plt.title("NNLD-SL-CeL", fontdict=font, bbox=dict(ec='pink', fc='orange'), verticalalignment='bottom')
                flag = flag + 1
            plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.tight_layout()

            plt.subplot(5, 6, index + 4)
            center_cnames1 = ['blue', 'red', 'green', "purple"]
            for Ward_point1, Ward_id1 in zip(Ward_data, np.array(Ward_label)):
                plt.scatter(Ward_point1[0], Ward_point1[1], c=center_cnames1[Ward_id1], s=dd, alpha=1)
            if flag == 5:
                plt.title("NNLD-SL-WL", fontdict=font, bbox=dict(ec='pink', fc='orange'), verticalalignment='bottom')
                flag = flag + 1
            plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.tight_layout()


            plt.subplot(5, 6, index + 5)
            center_cnames1 = ['blue', 'red', 'green', "purple"]
            for NN_point1, NN_id1 in zip(NN_data, np.array(NN_label)):
                plt.scatter(NN_point1[0], NN_point1[1], c=center_cnames1[NN_id1], s=dd, alpha=1)
            if flag == 6:
                plt.title("NNLD-SL-NNLICSM", fontdict=font, bbox=dict(ec='pink', fc='orange'), verticalalignment='bottom')
                flag = flag + 1
            plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
            plt.tight_layout()

            index = index+6
            zz = zz + 1
        ccc=ccc+1
        yflag = ccc
    plt.savefig(r'E:\Datasets\ICDM(0.05).jpeg', dpi=600)
    plt.show()










