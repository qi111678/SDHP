import numpy as np
import pandas as pd
import time
import math
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from NaN import nan_2
from NaN import nan_3

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
    percent = 0.02
    position = int(N * (N - 1) * percent)
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


def KROD_getDistanceMatrix(datas, a):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    dists = dists/(np.max(dists))
    list = np.argsort(dists)
    rows = np.argsort(list)
    rodist = (rows + rows.T)
    rodist = rodist / (np.max(rodist))
    KROD_value = rodist*np.exp((dists * dists) / (a * a))
    return KROD_value


if __name__ == "__main__":
    txtDF1 = pd.read_csv(r'E:\Datasets\D31.csv')
    col = txtDF1.shape[1]
    df = txtDF1.iloc[:, 0:(col - 1)]
    df_min_max = df.values

    nat_neighbor1 = nan_3.Natural_Neighbor()
    nat_neighbor1.load(r'E:\Datasets\D31_1.csv')
    nane1, temp_density1 = nat_neighbor1.algorithm()
    # -------------------------------------------------------------------------------------------
    dists = getDistanceMatrix(df_min_max)
    dc = select_dc(dists)
    rho_ruan = get_density(dists, dc)                                               # Equation(1)
    rho_ying=get_density(dists, dc, method="Gaussion")                              # Equation(2)
    # -------------------------------------------------------------------------------------------
    den_2017=[]
    for ele_2017 in range(len(df_min_max)):
        den_2017.append(np.sort(dists[ele_2017, :])[nane1])
    aver_knn=np.mean(den_2017)
    dc_2017=aver_knn+np.sqrt(np.sum((np.array(den_2017)-aver_knn)**2)/(len(df_min_max)-1))
    ture_den_2017=[]                                                                # Equation(3)
    for ele_2017 in range(len(df_min_max)):
        dis_2017=np.sort(dists[ele_2017, :])[1:nane1+1]
        ture_den_2017.append(np.sum(np.exp(-(dis_2017/dc_2017)**2)))
    # -------------------------------------------------------------------------------------------
    dists_2020 = KROD_getDistanceMatrix(df_min_max, 0.1)
    dc_2020 = select_dc(dists_2020)
    rho_KROD = get_density(dists_2020, dc_2020)                                      # Equation(7)
    # -------------------------------------------------------------------------------------------
    dists_2020_1 = KROD_getDistanceMatrix(df_min_max, 100)
    dc_2020_1 = select_dc(dists_2020_1)
    rho_KROD_1 = get_density(dists_2020_1, dc_2020_1)
    # -------------------------------------------------------------------------------------------
    ture_den_2016 = []                                                               # Equation(8)
    for ele_2016 in range(len(df_min_max)):
        dis_2016 = np.sort(dists[ele_2016, :])[1:nane1 + 1]
        ture_den_2016.append(np.exp(-np.mean(dis_2016*dis_2016)))
    # -------------------------------------------------------------------------------------------
    F_ture_den_2016 = []                                                             #  Equation(9)
    for Fele_2016 in range(len(df_min_max)):
        Fdis_2016 = np.sort(dists[Fele_2016, :])[1:nane1 + 1]
        F_ture_den_2016.append(np.sum(np.exp(-Fdis_2016)))
    # -------------------------------------------------------------------------------------------
    ture_den_2019 = []                                                               #  Equation(11)
    for nane_2019 in range(len(df_min_max)):
        Fdis_2016 = np.sort(dists[nane_2019, :])[1:nane1 + 1]
        ture_den_2019.append(nane1/np.sum(Fdis_2016))
    #-------------------------------------------------------------------------------------------
    ture_den_2022 = []                                                               #  Equation(12)
    for ele_2022 in range(len(df_min_max)):
        dis_2022 = np.sort(dists[ele_2022, :])[1:nane1 + 1]
        ture_den_2022.append(np.sum(np.exp(-dis_2022*dis_2022)))
    # -------------------------------------------------------------------------------------------
    nat_neighbor2 = nan_2.Natural_Neighbor()
    nat_neighbor2.load(r'E:\Datasets\D31_1.csv')
    nane2, temp_density2 = nat_neighbor2.algorithm()
    final_density1_our = []
    for i in temp_density2:
        final_density1_our.append(i / nane2)                                            # NN-LD
    # -------------------------------------------------------------------------------------------
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 48}
    font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': 46}
    hh = 32
    dd = 3.5
    plt.figure(figsize=(35, 14))
    plt.subplot(251)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=rho_ruan, marker='o', cmap='YlGnBu',lw=dd)
    plt.xlabel('(a) Equation(1)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(252)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=rho_ying, marker='o', cmap='YlGnBu',lw=dd)
    plt.xlabel('(b) Equation(2)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(253)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=np.array(ture_den_2017), marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(c) Equation(3)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(254)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=rho_KROD, marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(d_1) Equation(7)[0.1]', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------
    plt.subplot(255)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=rho_KROD_1, marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(d_2) Equation(7)[100]', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(256)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=np.array(ture_den_2016), marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(e) Equation(8)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(257)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=np.array(F_ture_den_2016), marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(f) Equation(9)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(258)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=np.array(ture_den_2019), marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(g) Equation(11)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(259)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=np.array(ture_den_2022), marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(h) Equation(12)', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    # -------------------------------------------------------------------------------------------
    plt.subplot(2,5,10)
    plt.scatter(df_min_max[:, 0], df_min_max[:, 1], c=np.array(final_density1_our), marker='o', cmap='YlGnBu', lw=dd)
    plt.xlabel('(i) NN-LD', font1)
    plt.xticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.yticks(fontsize=hh, family='Times New Roman', weight="bold")
    plt.tight_layout()
    plt.savefig(r'E:\Datasets\D31.jpeg', dpi=600)
    plt.show()