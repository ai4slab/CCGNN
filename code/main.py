import random

import numpy as np
import torch

from dataset import load_data
from model import CCGNN
from utils import plot_auc

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cpu')
data_dict = load_data()


def run():
    train_x = data_dict['train_x']
    train_y = torch.tensor(data_dict['train_y'], dtype=torch.float).to(device)
    test_x = data_dict['test_x']
    test_y_np = data_dict['test_y']
    test_y = torch.tensor(test_y_np, dtype=torch.float).to(device)
    model = CCGNN(data_dict, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    for epoch in range(210):
        model.train()
        optimizer.zero_grad()
        outs, loss = model(train_x, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        outs, _ = model(test_x, test_y)
        if epoch % 10 == 0:
            plot_auc(test_y_np, outs.cpu().detach().numpy(), epoch)


if __name__ == '__main__':
    run()
