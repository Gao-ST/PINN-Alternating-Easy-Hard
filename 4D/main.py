# 优化采样
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List
import os
import logging
from matplotlib_inline import backend_inline

# ================= File Config =====================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
backend_inline.set_matplotlib_formats("svg")
logger = logging.getLogger("ACWPINN_4D_Best")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
file_handler = logging.FileHandler("./log.txt")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

folder_path1 = "./model_save"
folder_path2 = "./figs"


def create_filedir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已成功创建", folder_path)
    else:
        print("文件夹已存在", folder_path)


create_filedir(folder_path1)
create_filedir(folder_path2)


# ======================= Model ============================
class mySin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class NN_mFF(nn.Module):
    def __init__(
        self, input_dim: int = 4, layers: List[int] = [100, 100, 100, 100, 1], sigma=10
    ):
        super().__init__()
        self.input_dim = input_dim
        self.W1 = nn.Parameter(torch.randn(1, layers[0] // 2), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(1, layers[0] // 2), requires_grad=True)
        self.W3 = nn.Parameter(torch.randn(1, layers[0] // 2), requires_grad=True)
        self.W4 = nn.Parameter(torch.randn(1, layers[0] // 2), requires_grad=True)
        self.layers = self.initialize_NN(layers)

    def initialize_NN(self, layers):
        layers_list = []
        for i in range(len(layers) - 2):
            layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            layers_list.append(mySin())
        layers_list.append(nn.Linear(4 * layers[-2], layers[-1]))
        return nn.Sequential(*layers_list)

    # Forward pass
    def forward(self, H):
        assert H.shape[1] == self.input_dim, "Wrong Input Dim."
        H1 = H[:, 0:1]
        H2 = H[:, 1:2]
        H3 = H[:, 2:3]
        H4 = H[:, 3:4]
        H1 = torch.cat([torch.sin(H1 @ self.W1), torch.cos(H1 @ self.W1)], dim=-1)
        H2 = torch.cat([torch.sin(H2 @ self.W2), torch.cos(H2 @ self.W2)], dim=-1)
        H3 = torch.cat([torch.sin(H3 @ self.W3), torch.cos(H3 @ self.W3)], dim=-1)
        H4 = torch.cat([torch.sin(H4 @ self.W4), torch.cos(H4 @ self.W4)], dim=-1)

        for layer in self.layers[:-1]:
            H1 = layer(H1)
            H2 = layer(H2)
            H3 = layer(H3)
            H4 = layer(H4)
        # H = H1 * H2 * H3 * H4
        H = torch.concat([H1, H2, H3, H4], -1)
        H = self.layers[-1](H)  # Dim = 1
        return H


# ============================= Equation Config ===========================
def DIFF(u, x, x_std):
    """
    计算 u(x1, x2) 在二维空间中的拉普拉斯算子 ∆u，支持输入归一化
    :param u: [N, 1] scalar-valued function
    :param x: [N, 2] input coordinates
    :param x_std: [2] 标准差张量（或列表），用于归一化偏导数
    :return: [N, 1] Laplacian ∆u
    """
    assert x.shape[1] == 4, "输入x应为二维向量"

    grads = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]

    lap = 0.0
    for i in range(4):
        grad_i = grads[:, i : i + 1] / x_std[i]  # 一阶导除以 std
        grad2_i = (
            torch.autograd.grad(
                grad_i,
                x,
                grad_outputs=torch.ones_like(grad_i),
                retain_graph=True,
                create_graph=True,
            )[0][:, i : i + 1]
            / x_std[i]
        )  # 二阶导也除以 std
        lap += grad2_i

    return lap.reshape(-1, 1)  # [N, 1]


def g(x):
    assert x.shape[1] == 4
    f = (
        (
            -16 * torch.pi**2 * torch.sin(4 * torch.pi * x[:, 0:1])
            - 36 * torch.pi**2 * torch.sin(6 * torch.pi * x[:, 1:2])
            - 64 * torch.pi**2 * torch.sin(8 * torch.pi * x[:, 2:3])
            - 250 * torch.pi**2 * torch.sin(50 * torch.pi * x[:, 3:4])
        )
        .detach()
        .data
    )  # First part source item

    return f.reshape(-1, 1)


def pde(u_pred, x_mean, x_std, x):
    u_xx = DIFF(u_pred, x, x_std)  # Laplace
    x_stds = torch.tensor(x_std, dtype=torch.float32).cuda()
    x_means = torch.tensor(x_mean, dtype=torch.float32).cuda()
    return g(x * x_stds + x_means) - u_xx


def _u(x):
    """
    输入:
        x: [N, 4] 张量，对应4维坐标
    输出:
        u: [N, 1] 张量，标量函数值
    """
    assert x.shape[1] == 4, "输入必须是4维空间点"

    u = (
        torch.sin(4 * math.pi * x[:, 0:1])
        + torch.sin(6 * math.pi * x[:, 1:2])
        + torch.sin(8 * math.pi * x[:, 2:3])
        + 0.1 * torch.sin(50 * math.pi * x[:, 3:4])
    )

    return u.reshape(-1, 1)  # [N, 1]


# ========================= Data Config ======================================
from pyDOE import lhs

N_f = 12800
x = lhs(4, N_f)
x_mean, x_std = x.mean(axis=0), x.std(axis=0)
x = torch.tensor((x - x_mean) / x_std, dtype=torch.float32, requires_grad=True).cuda()


def sample_4d_boundary_points(N):
    """
    从4维单位超立方体的边界 ∂Ω 中均匀采样N个点
    """
    B = []
    for i in range(4):
        for val in [0.0, 1.0]:
            x = torch.rand(N, 4)
            x[:, i] = val
            B.append(x)
    return torch.cat(B, dim=0)


Nb = 2400

# Boundary point with supervised learning as Vanilla PINN
xb = sample_4d_boundary_points(Nb)
u_b = _u(xb).cuda()
xb = torch.tensor((xb - x_mean) / x_std, dtype=torch.float32).cuda()

col_weights = torch.nn.Parameter(torch.ones((x.shape[0], 1)), requires_grad=True)
u_weights = torch.nn.Parameter(
    torch.ones((xb.shape[0], 1)) * torch.tensor([100]), requires_grad=True
)


# # ======================= Training Config ===================================nn
model = NN_mFF(input_dim=4, layers=[100, 100, 100, 100, 1]).cuda()

epochs = 100000
lr = 0.001
u_lr = 0.001
col_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_coll = torch.optim.Adam([col_weights], lr=col_lr)
optimizer_u = torch.optim.Adam([u_weights], lr=u_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2100, eta_min=1e-12, last_epoch=-1
)
best = 2


def train_StageI(
    data, xb, u_b, x_mean, x_std, col_weights, u_weights, pde, inner_epoch=10
):
    model.train()
    for k in range(inner_epoch):
        u_pred = model(data)
        f_u_pred = torch.abs(pde(u_pred, x_mean, x_std, data))
        u_b_pred = model(xb)
        b_loss = (torch.square((u_b_pred - u_b) * u_weights.cuda())).mean()
        f_loss = (torch.square(f_u_pred * col_weights.cuda())).mean()

        b_loss_weight = max(10, min(100 / (epoch + 1), 100))
        loss_value = f_loss + b_loss * b_loss_weight

        optimizer_coll.zero_grad()
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        loss_value.sum().backward()
        col_weights.grad = -col_weights.grad
        u_weights.grad = -u_weights.grad
        optimizer_coll.step()
        optimizer_u.step()
        optimizer.step()
    return model, loss_value, col_weights, u_weights


def train_StageII(
    data, xb, u_b, x_mean, x_std, col_weights, u_weights, pde, epoch, inner_epoch=300
):
    bro = (epoch + 1) % 300

    def select_rate(bro):
        rate = 0.5 + 0.99 * (bro) / 300
        if rate > 0.99:
            return 0.99
        else:
            return rate

    for k in range(inner_epoch):
        u_pred = model(data)

        f_u_pred = torch.abs(pde(u_pred, x_mean, x_std, data))

        if (bro) <= 00:
            loss_f, _ = torch.topk(
                f_u_pred, int(select_rate(bro) * len(f_u_pred)), dim=0, largest=False
            )
        else:
            loss_f = f_u_pred

        u_b_pred = model(xb)

        b_loss = (torch.square((u_b_pred - u_b))).mean()
        f_loss = (torch.square(loss_f)).mean()
        b_loss_weight = max(10, min(100 / (epoch + 1), 100))
        loss_value = f_loss + b_loss * b_loss_weight

        optimizer.zero_grad()
        loss_value.sum().backward()
        optimizer.step()
        scheduler.step()

    return model, loss_value, col_weights, u_weights


# ==================== Training ===========================
import time

# Testing point are random generated by `lhs`
ttime = 0
p = torch.tensor(lhs(4, 1001), dtype=torch.float32)
u_true = _u(p).detach().numpy().ravel()
p = (p - torch.tensor(x_mean, dtype=torch.float32)) / torch.tensor(
    x_std, dtype=torch.float32
)
for epoch in range(epochs):
    start = time.time()
    model, loss_value, col_weights, u_weights = train_StageI(
        x, xb, u_b, x_mean, x_std, col_weights, u_weights, pde, inner_epoch=50
    )
    model, loss_value, col_weights, u_weights = train_StageII(
        x, xb, u_b, x_mean, x_std, col_weights, u_weights, pde, epoch, inner_epoch=1
    )

    # This part can be deleted, we are using this
    # to verify whether our method can perfectly solve this pde?
    pred = model(p.cuda()).cpu().detach().numpy().ravel()
    rel2 = np.linalg.norm(pred - u_true, 2) / np.linalg.norm(u_true, 2)

    if rel2 < best:
        best = rel2
        torch.save(model.state_dict(), "./Best.pt")
        np.save("x_mean.npy", x_mean)
        np.save("x_std.npy", x_std)

    if (epoch + 1) % 1 == 0:
        txt = f"Epoch:[{epoch+1}/{epochs}], loss: {loss_value.cpu().detach().numpy()}, ReL2 Loss: {rel2.item()}. Total Time:{ttime}"
        logger.info(txt)
        ttime += time.time() - start
    # ======================= Plotting =================================

    if (epoch + 1) % 1000 == 0:
        torch.save(model.state_dict(), f"./model_save/Model_{epoch+1}.pt")
