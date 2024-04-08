import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = gnn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x1 = torch.relu(self.conv1(x, edge_index, edge_attr))
        x2 = torch.relu(self.conv2(x1, edge_index, edge_attr))
        return torch.relu(self.conv3(x2, edge_index, edge_attr))


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim=128, y_dim=128, hidden_size=128):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CCGNN(nn.Module):  # noqa
    def __init__(self, data_dict, device, embedding_size=128):
        super().__init__()
        self.xr = data_dict['XR'].to(device)
        self.xp = data_dict['XP'].to(device)
        self.x = torch.cat([self.xr, self.xp])
        self.feature_dim = self.x.shape[1]
        self.embedding_size = embedding_size
        self.num_rs = len(self.xr)
        self.num_ps = len(self.xp)
        self.rp_edge_index = data_dict['G'].to(device)
        self.rr_edge_index = data_dict['GR'].to(device)
        self.rr_edge_attr = data_dict['SR'].to(device)
        self.pp_edge_index = data_dict['GP'].to(device)
        self.pp_edge_attr = data_dict['SP'].to(device)

        self.rr_xr_proj = nn.Linear(self.feature_dim, self.embedding_size)
        self.rp_xr_proj = nn.Linear(self.feature_dim, self.embedding_size)
        self.rp_xp_proj = nn.Linear(self.feature_dim, self.embedding_size)
        self.pp_xp_proj = nn.Linear(self.feature_dim, self.embedding_size)
        nn.init.xavier_normal_(self.rr_xr_proj.weight)
        nn.init.xavier_normal_(self.rp_xr_proj.weight)
        nn.init.xavier_normal_(self.rp_xp_proj.weight)
        nn.init.xavier_normal_(self.pp_xp_proj.weight)

        self.rr_encoder = Encoder(embedding_size, embedding_size, embedding_size)
        self.rp_encoder = Encoder(embedding_size, embedding_size, embedding_size)
        self.pp_encoder = Encoder(embedding_size, embedding_size, embedding_size)

        self.r_at = Attention(embedding_size)
        self.p_at = Attention(embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()
        )
        self.i_mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()
        )
        self.alpha = 0.5
        self.gamma = 0.2
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, samples, labels):
        rr_xr = self.rr_xr_proj(self.xr)
        pp_xp = self.pp_xp_proj(self.xp)
        rp_xr = self.rp_xr_proj(self.xr)
        rp_xp = self.rp_xp_proj(self.xp)
        rp_x = torch.cat([rp_xr, rp_xp])

        rr_hr = self.rr_encoder(rr_xr, self.rr_edge_index, self.rr_edge_attr)
        rp_h = self.rp_encoder(rp_x, self.rp_edge_index, None)
        pp_hp = self.pp_encoder(pp_xp, self.pp_edge_index, self.pp_edge_attr)

        rp_hr, rp_hp = torch.split(rp_h, split_size_or_sections=[self.num_rs, self.num_ps])
        hr = torch.stack([rr_hr, rp_hr], dim=1)
        hp = torch.stack([pp_hp, rp_hp], dim=1)
        hr = self.r_at(hr)
        hp = self.p_at(hp)
        z = torch.cat([hr, hp])
        u = z[samples[:, 0]]
        v = z[samples[:, 1]]
        uv = torch.cat([u, v], dim=1)
        out = torch.squeeze(self.mlp(uv))
        pred_loss = self.criterion(out, labels)
        u_i = rp_h[samples[:, 0]]
        v_i = rp_h[samples[:, 1]]
        uv_i = torch.cat([u_i, v_i], dim=1)
        out_i = torch.squeeze(self.i_mlp(uv_i))
        pred_i_loss = self.criterion(out_i, labels)
        z_c = torch.cat([rr_hr, pp_hp])
        u_c = z_c[samples[:, 0]]
        v_c = z_c[samples[:, 1]]
        uv_c = torch.cat([u_c, v_c], dim=1)
        out_c = torch.squeeze(self.c_mlp(uv_c))
        pred_c_loss = self.criterion(out_c, labels)
        loss = pred_loss + self.alpha * (pred_i_loss + pred_c_loss)
        return out, loss, rr_hr, rp_hr, pp_hp, rp_hp
