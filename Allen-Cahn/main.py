import torch
from torch import nn
import numpy as np
import os
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_inline import backend_inline
import time

# 设置日志处理程序
import logging
from scipy.io import loadmat

# generate mesh for plotting
target_data = loadmat("./Allen_Cahn_0.001.mat")["u"].reshape(101, 201).T
logger = logging.getLogger("AllenCahn_Hardest")
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


def create_filedir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已成功创建", folder_path)
    else:
        print("文件夹已存在", folder_path)


create_filedir(folder_path1)


# --------------- Dataset ----------------------
from scipy.io import loadmat

lb = np.array([0.0])
ub = np.array([1.0])
rb = np.array([1.0])
lftb = np.array([-1.0])


N0 = 1200
N_b = 1200  # 25 per upper and lower boundary, so 50 total
N_f = 12800

x0 = lhs(1, N_b) * 2 - 1
tb = lhs(1, N_b)
x0 = np.concatenate([x0, np.array([[-1.0], [1.0]], dtype=np.float32)], 0)
tb = np.concatenate([tb, np.array([[0.0], [1.0]], dtype=np.float32)], 0)


X_lb = np.concatenate((x0, 0 * x0 + lb[0]), 1)  # lower boundary (x, 0)
X_ub = np.concatenate((x0, 0 * x0 + ub[0]), 1)  # upper boundary(x,1)
X_rb = np.concatenate((0 * tb + rb[0], tb), 1)  # right boundary (1, t)
X_lftb = np.concatenate((0 * tb + lftb[0], tb), 1)  # left boundary (-1,t)


data = torch.tensor(
    lhs(2, N_f) * np.array([2, 1]) - np.array([1, 0]),
    dtype=torch.float32,
).cuda()

xb = torch.tensor(np.concatenate([X_lb, X_rb, X_lftb], 0), dtype=torch.float32).cuda()
data.requires_grad = True
xb.requires_grad = True
x0 = torch.tensor(x0, dtype=torch.float32)
tb = torch.tensor(tb, dtype=torch.float32)
ub = torch.concat(
    [
        x0**2 * torch.cos(x0 * torch.pi),
        -torch.ones_like(tb),
        -torch.ones_like(tb),
    ],
    0,
).cuda()


col_weights = torch.nn.Parameter(torch.rand([data.shape[0], 1]), requires_grad=True)
u_weights = torch.nn.Parameter(
    torch.ones(ub.shape) * torch.tensor([int(1e2)]), requires_grad=True
)


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
lr = 1e-3
batch_size = data.shape[0] // 10


# # --------------- Equation Config ----------------------
eps = 0.001


def Partial_D(u, x):
    grad = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad[:, 0:1]
    u_t = grad[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    return u_t, u_xx


def pde(u, u_t, u_xx, eps=eps):
    return -u_t + u_xx * eps + 5 * u * (1 - u * u)


from torch.utils.data import TensorDataset, DataLoader

target = torch.zeros((len(data), 1))
dataset = TensorDataset(data, target)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def loss(data, xb, ub, col_weights, u_weights, epoch):
    u_pred = model(data)
    u_t, u_xx = Partial_D(u_pred, data)
    f_u_pred = pde(u_pred, u_t, u_xx, eps)
    u_b_pred = model(xb)
    mse_b_u = (torch.square((u_b_pred - ub) * u_weights.cuda())).mean()
    mse_f_u = (torch.square((f_u_pred) * col_weights.cuda())).mean()
    b_weights = max(10, min(100 / (epoch + 1), 100))  # 递减函数
    loss_value = mse_f_u + mse_b_u * b_weights
    return loss_value


epochs = 100000
lr = 0.001
col_lr = 0.001
u_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_coll = torch.optim.Adam([col_weights], lr=col_lr)
optimizer_u = torch.optim.Adam([u_weights], lr=u_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2100, eta_min=0, last_epoch=-1
)
best = 2
interval = 0


def train_PINN(
    data, xb, ub, col_weights, u_weights, Partial_D, pde, epoch, inner_epoch=1000
):
    model.train()
    for k in range(inner_epoch):
        u_pred = model(data)
        u_t, u_xx = Partial_D(u_pred, data)
        f_u_pred = pde(u_pred, u_t, u_xx)
        u_b_pred = model(xb)
        mse_b_u = (torch.square((u_b_pred - ub))).mean()
        mse_f_u = (torch.square(f_u_pred)).mean()
        b_weights = max(10, min(100 / (epoch + 1), 100))
        loss_value = mse_f_u + mse_b_u * b_weights
        optimizer.zero_grad()
        loss_value.sum().backward()
        optimizer.step()
        scheduler.step()

    return model, loss_value, col_weights, u_weights


def train_StageI(
    data, xb, ub, col_weights, u_weights, Partial_D, pde, loss, epoch, inner_epoch=1
):
    model.train()
    for k in range(inner_epoch):
        loss_value = loss(data, xb, ub, col_weights, u_weights, epoch)
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
    data, xb, ub, col_weights, u_weights, Partial_D, pde, epoch, inner_epoch=1
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
        u_pred = model(data)
        u_t, u_xx = Partial_D(u_pred, data)
        f_u_pred = pde(u_pred, u_t, u_xx)

        if (bro) <= 300:
            loss_f, _ = torch.topk(
                f_u_pred, int(select_rate(bro) * len(f_u_pred)), dim=0, largest=False
            )
        else:
            loss_f = f_u_pred

        u_b_pred = model(xb)
        mse_b_u = (torch.square((u_b_pred - ub))).mean()

        mse_f_u = (torch.square(loss_f)).mean()

        b_weights = max(10, min(100 / (epoch + 1), 100))
        loss_value = mse_f_u + mse_b_u * b_weights
        optimizer.zero_grad()
        loss_value.sum().backward()
        optimizer.step()

    return model, loss_value, col_weights, u_weights


p = np.linspace(-1, 1, 201)
q = np.linspace(0, 1, 201)
P, Q = np.meshgrid(p, q, indexing="ij")
datasets = np.concatenate([P.flatten()[:, None], Q.flatten()[:, None]], 1)
pq0 = np.concatenate([p.reshape(-1, 1), np.zeros(shape=(len(p), 1))], 1)
pqlft = np.concatenate([-np.ones(shape=(len(q), 1)), q.reshape(-1, 1)], 1)
pqr = np.concatenate([np.ones(shape=(len(q), 1)), q.reshape(-1, 1)], 1)

pqb = np.concatenate([pq0, pqlft, pqr], 0)
upqb = np.concatenate(
    [
        (p.reshape(-1, 1)) ** 2 * np.cos(np.pi * p.reshape(-1, 1)),
        -np.ones((len(pqlft), 1)),
        -np.ones((len(pqr), 1)),
    ],
    0,
)

test_xb = torch.tensor(pqb, dtype=torch.float32, requires_grad=True).cuda()
test_ub = torch.tensor(upqb, dtype=torch.float32, requires_grad=True).cuda()
test_data = torch.tensor(datasets, dtype=torch.float32, requires_grad=True).cuda()

for epoch in range(epochs):
    for X, y in trainloader:
        start = time.time()
        model, loss_value, col_weights, u_weights = train_StageI(
            data,
            xb,
            ub,
            col_weights,
            u_weights,
            Partial_D,
            pde,
            loss,
            epoch,
            inner_epoch=10,
        )
        model, loss_value, col_weights, u_weights = train_StageII(
            data, xb, ub, col_weights, u_weights, Partial_D, pde, epoch, inner_epoch=1
        )
    scheduler.step()

    u_test = model(test_data)
    u_b_pred_test = model(test_xb)
    u_t_tst, u_xx_tst = Partial_D(u_test, test_data)

    f_u_pred_test = pde(u_test, u_t_tst, u_xx_tst)

    mse_f_u = torch.square(f_u_pred_test).mean()
    mse_b_u = torch.square((u_b_pred_test - test_ub)).mean()

    Valid = mse_f_u + mse_b_u

    if Valid < best:
        best = Valid
        torch.save(model.state_dict(), "./Best_Allen.pt")

    if (epoch + 1) % 1 == 0:
        time_p = time.time() - start
        interval += time_p
        text = f"Epoch:[{epoch+1}/{epochs}],Loss:{loss_value}, Valid: {Valid},Time: {time_p}s,Total Time:{interval}s."
        logger.info(text)
        start = time.time()

    if (epoch + 1) % 1000 == 0:
        torch.save(model.state_dict(), f"./model_save/Model_{epoch+1}.pt")
