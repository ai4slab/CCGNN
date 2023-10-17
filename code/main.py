import random

import numpy as np
import torch

from dataset import load_data
from model import CCGNN, CLUBSample
from utils import plot_auc

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda:0')
data_dict = load_data(device)


def run():
    train_x = data_dict['train_x']
    train_y = torch.tensor(data_dict['train_y'], dtype=torch.float).to(device)
    test_x = data_dict['test_x']
    test_y_np = data_dict['test_y']
    test_y = torch.tensor(test_y_np, dtype=torch.float).to(device)
    model = CCGNN(data_dict, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    r_mi = CLUBSample().to(device)
    p_mi = CLUBSample().to(device)
    rmi_optimizer = torch.optim.Adam(r_mi.parameters(), lr=0.0001, weight_decay=5e-4)
    pmi_optimizer = torch.optim.Adam(p_mi.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(1, 1001):
        model.train()
        optimizer.zero_grad()
        out, loss, rr_hr, rp_hr, pp_hp, rp_hp = model(train_x, train_y)
        for _ in range(1):
            rmi_optimizer.zero_grad()
            rmi_learn_loss = r_mi.learning_loss(rr_hr, rp_hr)
            rmi_learn_loss.backward(retain_graph=True)
            rmi_optimizer.step()
        for _ in range(1):
            pmi_optimizer.zero_grad()
            pmi_learn_loss = p_mi.learning_loss(pp_hp, rp_hp)
            pmi_learn_loss.backward(retain_graph=True)
            pmi_optimizer.step()
        mim_loss = r_mi(rr_hr, rp_hr) + p_mi(pp_hp, rp_hp)
        loss += model.gamma * mim_loss
        loss.backward()
        optimizer.step()
        model.eval()
        outs, _, _, _, _, _ = model(test_x, test_y)
        if epoch % 50 == 0:
            plot_auc(test_y_np, outs.cpu().detach().numpy(), epoch)


if __name__ == '__main__':
    run()
