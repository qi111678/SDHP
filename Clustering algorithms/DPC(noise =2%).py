import numpy as np
import pandas as pd
import time
import math
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment


def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


def select_dc(dists):
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    percent = 2
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]
    return dc


def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if i == 0:
            continue
        index_higher_rho = index_rho[:i]
        deltas[index] = np.min(dists[index, index_higher_rho])

        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]

    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    for i, center in enumerate(centers):
        labs[center] = i

    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if labs[index] == -1:
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


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

    for xunhuan in range(10):
        start_1 = time.perf_counter()
        dists = getDistanceMatrix(df_min_max)
        dc = select_dc(dists)
        rho = get_density(dists, dc, method="Gaussion")
        deltas, nearest_neiber = get_deltas(dists, rho)
        centers = find_centers_K(rho, deltas, NC)

        labs = cluster_PD(rho, centers, nearest_neiber)
        df_label = pd.DataFrame(labs.tolist(), columns=['label'])
        df_data = pd.DataFrame(df_min_max.tolist())
        D_data_result = pd.concat([df_data, df_label], axis=1)
        end_1 = time.perf_counter()
        time_1 = end_1 - start_1

        algorithm_consume_time.append(time_1)
        Sil.append(metrics.silhouette_score(df_min_max, labs, metric='euclidean'))
        CH.append(metrics.calinski_harabasz_score(df_min_max, labs))
        DB.append(metrics.davies_bouldin_score(df_min_max, labs))
        ARI.append(metrics.adjusted_rand_score(true_value_1, labs))
        AMI.append(metrics.adjusted_mutual_info_score(true_value_1, labs))
        HOM.append(metrics.homogeneity_score(true_value_1, labs))
        COM.append(metrics.completeness_score(true_value_1, labs))
        V.append(metrics.v_measure_score(true_value_1, labs))
        FMI.append(metrics.fowlkes_mallows_score(true_value_1, labs))
        ACC.append(acc(true_value_1, labs))
        NMI.append(metrics.normalized_mutual_info_score(true_value_1, labs))

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
                          columns=['Sil', 'CH', 'DB', 'ARI', 'AMI', 'HOM', 'COM', 'V', 'FMI', "ACC", "NMI",  'runtime'])
    store1.to_csv(r'E:\Datasets\DPC_D31(noise=2%).csv', index=False)