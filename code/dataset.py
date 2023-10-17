import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_undirected


def load_data(device):
    G = np.load(f'../data/G.npy')
    XR = np.load(f'../data/XR.npy')
    XP = np.load(f'../data/XP.npy')
    GR = np.load(f'../data/GR.npy')
    SR = np.load(f'../data/SR.npy')
    GP = np.load(f'../data/GP.npy')
    SP = np.load(f'../data/SP.npy')

    pos_index = np.where(G[:, 2] == 1)
    pos = G[pos_index][:, :2]
    pos_y = G[pos_index][:, 2]
    neg_index = np.where(G[:, 2] == 0)
    neg = G[neg_index][:, :2]
    neg_y = G[neg_index][:, 2]

    train_pos_x, test_pos_x, train_pos_y, test_pos_y = train_test_split(pos, pos_y, test_size=0.2)
    train_neg_x, test_neg_x, train_neg_y, test_neg_y = train_test_split(neg, neg_y, test_size=0.2)

    G = torch.tensor(train_pos_x.T, dtype=torch.long)
    G = to_undirected(G).to(device)
    XR = torch.tensor(XR, dtype=torch.float).to(device)
    XP = torch.tensor(XP, dtype=torch.float).to(device)
    GR = torch.tensor(GR.T, dtype=torch.long).to(device)
    SR = torch.tensor(SR, dtype=torch.float).to(device)
    GP = torch.tensor(GP.T, dtype=torch.long).to(device)
    SP = torch.tensor(SP, dtype=torch.float).to(device)
    train_x = np.concatenate([train_pos_x, train_neg_x])
    train_y = np.concatenate([train_pos_y, train_neg_y])
    test_x = np.concatenate([test_pos_x, test_neg_x])
    test_y = np.concatenate([test_pos_y, test_neg_y])

    num_lncRNAs = len(set(pos[:, 0]))
    num_proteins = len(set(pos[:, 1]))
    print(
        f'Loading data successfully, this dataset containing {num_lncRNAs} lncRNAs, {num_proteins} proteins, '
        f'{len(pos)} positive samples, and {len(neg)} negative samples.')
    return {
        'G': G, 'XR': XR, 'XP': XP, 'GR': GR, 'SR': SR, 'GP': GP, 'SP': SP, 'train_x': train_x, 'train_y': train_y,
        'test_x': test_x, 'test_y': test_y
    }


if __name__ == '__main__':
    load_data('cuda:0')
