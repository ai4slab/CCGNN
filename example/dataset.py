import numpy as np
import torch
from torch_geometric.utils import to_undirected


def load_data(device):
    MG = np.load(f'MTI_data/knn_graphs/MM_graph.npz', allow_pickle=True)
    GR = MG['edge']
    SR = MG['weight']
    TG = np.load(f'MTI_data/knn_graphs/TT_graph.npz', allow_pickle=True)
    GP = TG['edge']
    SP = TG['weight']
    XR = np.load(f'MTI_data/MiRNA_feature.npy')
    XP = np.load(f'MTI_data/Target_feature.npy')

    train_pos_x = np.load(f'MTI_data/splits/train_pos.npy')
    train_neg_x = np.load(f'MTI_data/splits/train_neg.npy')
    test_pos_x = np.load(f'MTI_data/splits/test_pos.npy')
    test_neg_x = np.load(f'MTI_data/splits/test_neg.npy')

    G = torch.tensor(train_pos_x.T, dtype=torch.long)
    G = to_undirected(G).to(device)
    XR = torch.tensor(XR, dtype=torch.float).to(device)
    XP = torch.tensor(XP, dtype=torch.float).to(device)
    GR = torch.tensor(GR.T, dtype=torch.long).to(device)
    SR = torch.tensor(SR, dtype=torch.float).to(device)
    GP = torch.tensor(GP.T, dtype=torch.long).to(device)
    SP = torch.tensor(SP, dtype=torch.float).to(device)

    train_x = np.concatenate([train_pos_x, train_neg_x])
    train_y = np.concatenate([np.ones(len(train_pos_x)), np.zeros(len(train_neg_x))])
    test_x = np.concatenate([test_pos_x, test_neg_x])
    test_y = np.concatenate([np.ones(len(test_pos_x)), np.zeros(len(test_neg_x))])

    return {
        'G': G, 'XR': XR, 'XP': XP, 'GR': GR, 'SR': SR, 'GP': GP, 'SP': SP, 'train_x': train_x, 'train_y': train_y,
        'test_x': test_x, 'test_y': test_y
    }


if __name__ == '__main__':
    load_data('cuda:0')
