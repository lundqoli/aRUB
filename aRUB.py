import torch.nn as nn
import torch
import torch.nn.functional as F

class aRUB(nn.Module):

    def __init__(self,epsilon,n_classes,device, norm="max"):
        super().__init__()

        self.epsilon = epsilon
        self.n_classes = n_classes
        self.norm = norm
        self.device = device
        self.create_grads = True

    def eval(self):
        self.create_grads = False

    def train(self):
        self.create_grads = True

    def forward(self, y, x, net):
        n_classes = self.n_classes
        n_batch = x.shape[0]
        epsilon = self.epsilon
        device = self.device

        x.requires_grad = True

        z = net(x)
        z = z.to(device)
        ones = torch.ones(n_classes).to(device)
        ones.requires_grad = True
        ck = torch.zeros((n_batch, n_classes, n_classes)).to(device)
        ck[:] = torch.diag(ones)

        for i in range(0, n_batch):
            ck[i, :, y[i]] = ck[i, :, y[i]] - ones

        z_expanded = z.unsqueeze(2)
        ez = torch.matmul(ck, z_expanded)
        ez = ez.squeeze(2)

        grads = []
        for i in range(0, n_classes):
            grads.append(torch.autograd.grad(ez[:, i], x,create_graph = True, grad_outputs=torch.ones_like(ez[:, i]))[0])

        stacked = torch.stack(grads)
        stacked_reshaped = stacked.permute((1, 0, 2, 3, 4))
        stacked_flatten = torch.flatten(stacked_reshaped, start_dim=2)

        if self.norm == "L1":
            xnorm = torch.linalg.norm(stacked_flatten, dim=2, ord=1)
        elif self.norm=="L2":
            xnorm = torch.linalg.norm(stacked_flatten, dim=2, ord=2)
        elif self.norm=="Linf":
            xnorm = torch.linalg.norm(stacked_flatten, dim=2, ord=float("inf"))

        exp = torch.exp(ez + epsilon*xnorm)
        exp_sum = exp.sum(1)

        log_exp_sum = torch.log(exp_sum)
        loss = log_exp_sum.mean()

        if self.create_grads:
            return loss, net
        else:
            return loss.detach()