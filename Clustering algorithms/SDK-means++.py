import numpy as np
import time
import math
import random
import copy
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
import pandas as pd


def get_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


def center_init(k, X):
    n_samples, n_features = X.shape
    centers = np.zeros((k, n_features))
    selected_centers_index = []
    for i in range(k):
        sel_index = random.choice(list(set(range(n_samples)) - set(selected_centers_index)))
        centers[i] = X[sel_index]
        selected_centers_index.append(sel_index)
    return centers


def closest_center(sample, centers):
    closest_i = 0
    closest_dist = float('inf')
    for i, c in enumerate(centers):
        distance = get_distance(sample, c)
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i


def create_clusters(centers, k, X):
    clusters = [[] for _ in range(k)]
    clusters_samples = [[] for _ in range(k)]
    for sample_i, sample in enumerate(X):
        center_i = closest_center(sample, centers)
        clusters[center_i].append(sample_i)
        clusters_samples[center_i].append(sample.tolist())
    return clusters, clusters_samples


def calculate_new_centers(clusters, k, X):
    n_samples, n_features = X.shape
    centers = np.zeros((k, n_features))
    for i, cluster in enumerate(clusters):
        new_center = np.mean(X[cluster], axis=0)
        centers[i] = new_center
    return centers


def get_cluster_labels(clusters, X):
    y_pred = np.zeros(np.shape(X)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred


def Mykmeans(X, k, max_iterations, init, initial_center):
    if init == 'kmeans':
        centers = center_init(k, X)
    else:
        centers = get_kmeansplus_centers(X, k, initial_center)
    for _ in range(max_iterations):
        clusters, clusters_samples = create_clusters(centers, k, X)
        pre_centers = centers
        new_centers = calculate_new_centers(clusters, k, X)
        centers = new_centers
        new_centers_1 = new_centers.tolist()

        pre_centers_1 = []
        for arr in range(len(pre_centers)):
            pre_centers_1.append(pre_centers[arr].tolist())

        new_centers_1.sort()
        pre_centers_1.sort()

        if new_centers_1 == pre_centers_1:
            times_my = _ + 1
            break
    return get_cluster_labels(clusters, X), clusters_samples, times_my


def euler_distance(point1: list, point2: list) -> float:
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def get_kmeansplus_centers(X, k, initial_center):
    cluster_centers = []
    cluster_centers.append(initial_center)
    temp_X = copy.deepcopy(X)
    for _ in range(1, k):
        temp_X_list = temp_X.tolist()
        for every_ele in temp_X_list:
            if every_ele == cluster_centers[-1].tolist():
                temp_X_list.remove(every_ele)
            else:
                continue
        temp_X = np.array(temp_X_list)
        d = [0 for _ in range(len(temp_X))]
        for ii, point in enumerate(temp_X):
            temp_sum = 0
            for jj, center_point in enumerate(cluster_centers):
                d1 = euler_distance(point, center_point)
                temp_sum += d1
            d[ii] = temp_sum
        cluster_centers.append(temp_X[d.index(max(d))])
    return cluster_centers


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
    temp1 = []

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
        true_value_1 = np.array(true_value_1)

        kind1 = []
        for kin in range(txtDF1.shape[0]):
            kind1.append(final_true_label1.values.tolist()[kin][0])

        NC = len(set(kind1))

        start_1 = time.perf_counter()
        sum_distance = 0
        for dc, dc_point in enumerate(df_min_max):
            for dc_1, dc_point_1 in enumerate(df_min_max):
                temp_dis = euler_distance(dc_point, dc_point_1)
                sum_distance += temp_dis

        average_distance_1 = sum_distance / 2
        average_distance = average_distance_1 / ((len(df_min_max) ** 2 - len(df_min_max)) / 2)

        final_dc = average_distance / 2

        density_count = [0 for _ in range(len(df_min_max))]
        for dc, dc_point in enumerate(df_min_max):
            for dc_1, dc_point_1 in enumerate(df_min_max):
                if (euler_distance(dc_point, dc_point_1) < final_dc):
                    density_count[dc] += 1

        first_center = df_min_max[density_count.index(max(density_count))]
        labels1, groups, times_5 = Mykmeans(df_min_max, k=NC, max_iterations=10000000000000000000000, init='kmeans++', initial_center=first_center)

        end_1 = time.perf_counter()

        labels = []
        for i1 in range(len(labels1)):
            labels.append(int(labels1[i1]))
        labels=np.array(labels)
        time_1 = end_1 - start_1

        distribution_value = []
        for ii in range(NC):
            distribution_value.append(len(groups[ii]))

        algorithm_consume_time.append(time_1)
        # ------------------------------------------------------------------
        df_label = pd.DataFrame(labels, columns=['label'])
        df_data = pd.DataFrame(df_min_max.tolist())
        D_data_result = pd.concat([df_data, df_label], axis=1)

        Sil.append(metrics.silhouette_score(df_min_max, labels, metric='euclidean'))
        CH.append(metrics.calinski_harabasz_score(df_min_max, labels))
        DB.append(metrics.davies_bouldin_score(df_min_max, labels))
        ARI.append(metrics.adjusted_rand_score(true_value_1, labels))
        AMI.append(metrics.adjusted_mutual_info_score(true_value_1, labels))
        HOM.append(metrics.homogeneity_score(true_value_1, labels))
        COM.append(metrics.completeness_score(true_value_1, labels))
        V.append(metrics.v_measure_score(true_value_1, labels))
        FMI.append(metrics.fowlkes_mallows_score(true_value_1, labels))
        ACC.append(acc(true_value_1, labels))
        NMI.append(metrics.normalized_mutual_info_score(true_value_1, labels))

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
    store1.to_csv(r'E:\Datasets\SDK-D31.csv', index=False)