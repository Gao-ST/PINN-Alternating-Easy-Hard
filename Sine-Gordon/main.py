import torch
from torch import nn
import numpy as np
import os
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_inline import backend_inline
import time
from torch.utils.data import TensorDataset, DataLoader

# 设置日志处理程序
import logging
from scipy.io import loadmat

# generate mesh for plotting
logger = logging.getLogger("SineGordon_ACW_1e-5Target")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.FileHandler("./log.txt")
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)


backend_inline.set_matplotlib_formats("svg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --------------- 生成数据的检验 ----------------------

import os

folder_path1 = "./model_save"
# folder_path2 = "./figs"
# folder_path3 = "./weights"


def create_filedir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已成功创建", folder_path)
    else:
        print("文件夹已存在", folder_path)


create_filedir(folder_path1)
# create_filedir(folder_path2)
# create_filedir(folder_path3)


def _u(x):
    return np.sin(x[:, 0:1] * x[:, 1:2]) * (1 - np.cos(x[:, 0:1] ** 2 + x[:, 1:2] ** 2))


def Laplace(X, u):
    grad = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grad[:, 0:1]
    u_y = grad[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2]
    return u_xx + u_yy


class MLP(nn.Module):
    def __init__(self, lst: list, act: callable = nn.Tanh):
        super().__init__()
        depth = len(lst) - 1
        layer = []
        for i in range(depth - 1):
            linear = nn.Linear(lst[i], lst[i + 1])
            layer.append(linear)
            layer.append(act())
        layer.append(nn.Linear(lst[-2], lst[-1]))
        self.layer = nn.Sequential(*layer)

    def forward(self, X):
        return self.layer(X)


model = MLP([2] + [50] * 4 + [1]).cuda()


def pde(X):
    u_pred = model(X)
    u_true = torch.sin(X[:, 0:1] * X[:, 1:2]) * (
        1 - torch.cos(X[:, 0:1] ** 2 + X[:, 1:2] ** 2)
    )

    laplace_u_pred = Laplace(X, u_pred)
    laplace_u_true = Laplace(X, u_true)
    return laplace_u_pred + torch.sin(u_pred) - laplace_u_true - torch.sin(u_true)


def loss(data, xb, u_b, col_weights, u_weights, epoch):
    f_u_pred = pde(data)
    u_b_pred = model(xb)
    mse_b_u = (torch.square((u_b_pred - u_b) * u_weights.cuda())).mean()
    mse_f_u = (torch.square((f_u_pred) * col_weights.cuda())).mean()
    b_weights = max(10, min(100 / (epoch + 1), 100))  # 递减函数
    loss_value = mse_f_u + mse_b_u * b_weights
    return loss_value


lb = np.array([-4.0])
ub = np.array([4.0])
rb = np.array([4.0])
lftb = np.array([-4.0])


N = 120
nx, ny = (N, N)
x = np.linspace(lftb[0], rb[0], nx, endpoint=True)
y = np.linspace(lb[0], ub[0], ny, endpoint=True)

xv, yv = np.meshgrid(x, y, indexing="ij")

data = np.concatenate([xv.flatten()[:, None], yv.flatten()[:, None]], 1)
mask = np.logical_or.reduce(
    [
        data[:, 0] == lftb[0],
        data[:, 0] == rb[0],
        data[:, 1] == lb[0],
        data[:, 1] == ub[0],
    ]
)
xb = data[mask].copy()
u_b = _u(xb).reshape(-1, 1)


data = torch.tensor(data, dtype=torch.float32, requires_grad=True).cuda()
xb = torch.tensor(xb, dtype=torch.float32, requires_grad=True).cuda()
u_b = torch.tensor(u_b, dtype=torch.float32, requires_grad=True).cuda()

col_weights = torch.nn.Parameter(torch.ones([data.shape[0], 1]), requires_grad=True)
u_weights = torch.nn.Parameter(
    torch.ones(u_b.shape) * torch.tensor([int(1e2)]), requires_grad=True
)
eps = 0.001
epochs = 100000
lr = 0.001
col_lr = 0.001
u_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_coll = torch.optim.Adam([col_weights], lr=col_lr)
optimizer_u = torch.optim.Adam([u_weights], lr=u_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2100, eta_min=1e-9, last_epoch=-1
)
best = 2
interval = 0


def train_StageI(
    data, xb, u_b, col_weights, u_weights, Laplace, pde, loss, epoch, inner_epoch=100
):
    model.train()
    for k in range(inner_epoch):
        loss_value = loss(data, xb, u_b, col_weights, u_weights, epoch)
        optimizer_coll.zero_grad()
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        loss_value.sum().backward()
        col_weights.grad = -col_weights.grad
        u_weights.grad = -u_weights.grad
        optimizer_u.step()
        optimizer_coll.step()
        optimizer.step()
    return model, loss_value, col_weights, u_weights


def train_StageII(
    data,
    xb,
    u_b,
    col_weights,
    u_weights,
    loss_value,
    Laplace,
    pde,
    loss,
    epoch,
    inner_epoch=1000,
):
    model.train()

    if (epoch + 1) <= 300:
        bro = epoch + 1
    else:
        bro = (epoch + 1) % 300

    def select_rate(bro):
        rate = 0.5 + 0.99 * (bro) / 300
        if rate > 0.99:
            return 0.99
        else:
            return rate

    for k in range(inner_epoch):
        f_u_pred = torch.abs(pde(data))
        if (bro) <= 00:
            loss_f, _ = torch.topk(
                f_u_pred, int(select_rate(bro) * len(f_u_pred)), dim=0, largest=False
            )
        else:
            loss_f = f_u_pred

        u_b_pred = model(xb)
        mse_b_u = (torch.square((u_b_pred - u_b))).mean()

        mse_f_u = (torch.square(loss_f)).mean()

        b_weights = max(10, min(100 / (epoch + 1), 100))
        loss_value = mse_f_u + mse_b_u * b_weights
        optimizer.zero_grad()
        loss_value.sum().backward()
        optimizer.step()

    return model, loss_value, col_weights, u_weights


R, S = np.meshgrid(
    np.linspace(lftb[0], rb[0], N), np.linspace(lb[0], ub[0], N), indexing="ij"
)
test_data = np.concatenate([R.flatten()[:, None], S.flatten()[:, None]], 1)
truth = _u(test_data).ravel()

test_data_tensor = torch.tensor(test_data, dtype=torch.float32).cuda()
for epoch in range(epochs):
    start = time.time()
    model, loss_value, col_weights, u_weights = train_StageI(
        data,
        xb,
        u_b,
        col_weights,
        u_weights,
        Laplace,
        pde,
        loss,
        epoch,
        inner_epoch=10,
    )
    model, loss_value, col_weights, u_weights = train_StageII(
        data,
        xb,
        u_b,
        col_weights,
        u_weights,
        loss_value,
        Laplace,
        pde,
        loss,
        epoch,
        inner_epoch=0,
    )
    scheduler.step()
    predd = model(test_data_tensor).cpu().detach().numpy().ravel()
    rel2 = np.linalg.norm(predd - truth, 2) / np.linalg.norm(truth, 2)
    #     # ===================================
    if rel2 < best:
        best = rel2
        torch.save(model.state_dict(), "./Best_SG.pt")

    if (epoch + 1) % 1 == 0:
        time_p = time.time() - start
        interval += time_p
        text = f"Epoch:[{epoch+1}/{epochs}],Loss:{loss_value}, Time: {time_p}s, Total Time:{interval}s, ReL2:{rel2}."
        logger.info(text)

    if (epoch + 1) % 1000 == 0:
        torch.save(model.state_dict(), f"./model_save/Model_{epoch+1}.pt")
