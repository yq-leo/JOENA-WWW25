import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(input_dim + hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        pos_emd1 = torch.cat([x1, self.act(self.lin1(x1))], dim=1)
        pos_emd2 = torch.cat([x2, self.act(self.lin1(x2))], dim=1)
        pos_emd1 = self.lin2(pos_emd1)
        pos_emd2 = self.lin2(pos_emd2)
        pos_emd1 = F.normalize(pos_emd1, p=2, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=2, dim=1)
        return pos_emd1, pos_emd2


class FusedGWLoss(torch.nn.Module):
    def __init__(self, G1_tg, G2_tg, anchor1, anchor2, gw_weight=20, gamma_p=1e-2, init_threshold_lambda=1, in_iter=5,
                 out_iter=10, total_epochs=250):
        super().__init__()
        self.device = G1_tg.x.device
        self.gw_weight = gw_weight
        self.gamma_p = gamma_p
        self.in_iter = in_iter
        self.out_iter = out_iter
        self.total_epochs = total_epochs

        self.n1, self.n2 = G1_tg.num_nodes, G2_tg.num_nodes
        self.threshold_lambda = init_threshold_lambda / (self.n1 * self.n2)
        self.adj1, self.adj2 = G1_tg.adj, G2_tg.adj
        self.H = torch.ones(self.n1, self.n2).to(torch.float64).to(self.device)
        self.H[anchor1, anchor2] = 0

    def forward(self, out1, out2):
        inter_c = torch.exp(-(out1 @ out2.T))
        intra_c1 = torch.exp(-(out1 @ out1.T)) * self.adj1
        intra_c2 = torch.exp(-(out2 @ out2.T)) * self.adj2
        with torch.no_grad():
            s = sinkhorn_stable(inter_c, intra_c1, intra_c2,
                                gw_weight=self.gw_weight,
                                gamma_p=self.gamma_p,
                                threshold_lambda=self.threshold_lambda,
                                in_iter=self.in_iter,
                                out_iter=self.out_iter,
                                device=self.device)
            self.threshold_lambda = 0.05 * self.update_lambda(inter_c, intra_c1, intra_c2, s) + 0.95 * self.threshold_lambda

        s_hat = s - self.threshold_lambda

        # Wasserstein Loss
        w_loss = torch.sum(inter_c * s_hat)

        # Gromov-Wasserstein Loss
        a = torch.sum(s_hat, dim=1)
        b = torch.sum(s_hat, dim=0)
        gw_loss = torch.sum(
            (intra_c1 ** 2 @ a.view(-1, 1) @ torch.ones((1, self.n2)).to(torch.float64).to(self.device) +
             torch.ones((self.n1, 1)).to(torch.float64).to(self.device) @ b.view(1, -1) @ intra_c2 ** 2 -
             2 * intra_c1 @ s_hat @ intra_c2.T) * s_hat)

        loss = w_loss + self.gw_weight * gw_loss + 20
        return loss, s, self.threshold_lambda

    def update_lambda(self, inter_c, intra_c1, intra_c2, s):
        k1 = torch.sum(inter_c)

        one_mat = torch.ones(self.n1, self.n2).to(torch.float64)
        mid = intra_c1 ** 2 @ one_mat * self.n2 + one_mat @ intra_c2 ** 2 * self.n1 - 2 * intra_c1 @ one_mat @ intra_c2.T
        k2 = torch.sum(mid * s)
        k3 = torch.sum(mid)

        return (k1 + 2 * self.gw_weight * k2) / (2 * self.gw_weight * k3)


def sinkhorn_stable(inter_c, intra_c1, intra_c2, threshold_lambda, in_iter=5, out_iter=10, gw_weight=20, gamma_p=1e-2,
                    device='cpu'):
    n1, n2 = inter_c.shape
    # marginal distribution
    a = torch.ones(n1).to(torch.float64).to(device) / n1
    b = torch.ones(n2).to(torch.float64).to(device) / n2
    # lagrange multiplier
    f = torch.ones(n1).to(torch.float64).to(device) / n1
    g = torch.ones(n2).to(torch.float64).to(device) / n2
    # transport plan
    s = torch.ones((n1, n2)).to(torch.float64).to(device) / (n1 * n2)

    def soft_min_row(z_in, eps):
        hard_min = torch.min(z_in, dim=1, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=1, keepdim=True))
        return soft_min.squeeze(-1)

    def soft_min_col(z_in, eps):
        hard_min = torch.min(z_in, dim=0, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=0, keepdim=True))
        return soft_min.squeeze(0)

    for i in range(out_iter):
        a_hat = torch.sum(s - threshold_lambda, dim=1)
        b_hat = torch.sum(s - threshold_lambda, dim=0)
        temp = (intra_c1 ** 2 @ a_hat.view(-1, 1) @ torch.ones((1, n2)).to(torch.float64).to(device) +
                torch.ones((n1, 1)).to(torch.float64).to(device) @ b_hat.view(1, -1) @ intra_c2 ** 2)
        L = temp - 2 * intra_c1 @ (s - threshold_lambda) @ intra_c2.T
        cost = inter_c + gw_weight * L

        Q = cost
        for j in range(in_iter):
            # log-sum-exp stabilization
            f = soft_min_row(Q - g.view(1, -1), gamma_p) + gamma_p * torch.log(a)
            g = soft_min_col(Q - f.view(-1, 1), gamma_p) + gamma_p * torch.log(b)
        s = 0.05 * s + 0.95 * torch.exp((f.view(-1, 1) + g.view(-1, 1).T - Q) / gamma_p)

    return s
