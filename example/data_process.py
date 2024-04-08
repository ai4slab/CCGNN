import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

seed = 2024
random.seed(seed)
np.random.seed(seed)


def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)  # noqa
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


if __name__ == '__main__':
    data_pos_df = pd.read_csv('MTI_data/MTI_pos.csv', header=None, sep='\t')
    data_pos = data_pos_df.values
    data_neg_df = pd.read_csv('MTI_data/MTI_neg.csv', header=None, sep='\t')
    data_neg = data_neg_df.values

    MiRNA_df = pd.read_csv('MTI_data/MiRNA.csv', header=None, sep='\t')
    Target_df = pd.read_csv('MTI_data/Target.csv', header=None, sep='\t')
    num_miRNAs = len(MiRNA_df)
    num_targets = len(Target_df)

    data_pos[:, 1] += num_miRNAs
    data_neg[:, 1] += num_miRNAs

    train_pos, test_pos = train_test_split(data_pos, test_size=0.2, random_state=seed)
    train_neg, test_neg = train_test_split(data_neg, test_size=0.2, random_state=seed)

    np.save(f'MTI_data/splits/train_pos.npy', train_pos)
    np.save(f'MTI_data/splits/test_pos.npy', test_pos)
    np.save(f'MTI_data/splits/train_neg.npy', train_neg)
    np.save(f'MTI_data/splits/test_neg.npy', test_neg)

    M_feature = np.load('MTI_data/MiRNA_feature.npy')
    T_feature = np.load('MTI_data/Target_feature.npy')
    MM_sim_mat = cosine_similarity(M_feature)
    TT_sim_mat = cosine_similarity(T_feature)
    MM_mat = k_matrix(MM_sim_mat)
    TT_mat = k_matrix(TT_sim_mat)
    MM_graph_edges = np.stack(np.where(MM_mat > 0), axis=1)
    MM_graph_weights = MM_mat[MM_graph_edges[:, 0], MM_graph_edges[:, 1]]
    TT_graph_edges = np.stack(np.where(TT_mat > 0), axis=1)
    TT_graph_weights = TT_mat[TT_graph_edges[:, 0], TT_graph_edges[:, 1]]

    np.savez('MTI_data/knn_graphs/MM_graph.npz', edge=MM_graph_edges, weight=MM_graph_weights)
    np.savez('MTI_data/knn_graphs/TT_graph.npz', edge=TT_graph_edges, weight=TT_graph_weights)
    print(f'MTI dataset: {num_miRNAs} miRNAs, {num_targets} targets, {len(data_pos)} MTIs.')
