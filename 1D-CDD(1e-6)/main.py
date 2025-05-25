import torch
from torch import nn
import numpy as np
import os
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_inline import backend_inline
from torch.utils.data import TensorDataset, DataLoader
import time
from typing import List
import math

# 设置日志处理程序
import logging

logger = logging.getLogger("1e-6_2501")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
filehandler = logging.FileHandler("./log.txt")
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)

logger.addHandler(filehandler)
logger.addHandler(console)


backend_inline.set_matplotlib_formats("svg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
folder_path1 = "./model_save"
folder_path2 = "./figs"
folder_path3 = "./weights"


def create_filedir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Create", folder_path)
    else:
        print("Exist", folder_path)


create_filedir(folder_path1)
create_filedir(folder_path2)
create_filedir(folder_path3)
# --------------- Dataset ----------------------
lb = np.array([0.0])
ub = np.array([1.0])

N_f = 2501
_eps = 1e-6

col_weights = torch.ones([N_f, 1])
col_weights = torch.nn.Parameter(col_weights, requires_grad=True)
u_weights = torch.nn.Parameter(
    torch.ones([2, 1]) * torch.tensor([1e4]), requires_grad=True
)
X_f = lhs(1, N_f)

data = torch.tensor(X_f[:, 0:1], dtype=torch.float32).cuda()
x_lb = torch.tensor(lb, dtype=torch.float32).cuda()
x_ub = torch.tensor(ub, dtype=torch.float32).cuda()


data.requires_grad = True
x_lb.requires_grad = True
x_ub.requires_grad = True

trainloader = DataLoader(
    TensorDataset(data, torch.zeros((len(data), 1))), batch_size=128, shuffle=True
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


model = MLP([1] + [50] * 4 + [1]).cuda()
model.load_state_dict(torch.load("Pre-Trained.pt"))
lr = 1e-3
batch_size = 128


def _u(x):
    return torch.cos(math.pi * x / 2) * (1 - torch.exp(-2 * x / _eps))


def _u_grad_first_order(x):
    tmp = torch.exp(-2 * x / _eps)
    return (
        -torch.sin(math.pi * x / 2) * math.pi / 2 * (1 - tmp)
        + torch.cos(math.pi * x / 2) * tmp * 2 / _eps
    )


def _u_grad_second_order(x):
    tmp = torch.exp(-2 * x / _eps)
    part1 = math.pi**2 / 4 * (1 - tmp) + tmp * 4 / (_eps**2)
    part2 = 2 * math.pi / _eps * tmp
    return -torch.cos(math.pi / 2 * x) * part1 - torch.sin(math.pi / 2 * x) * part2


def _f_right(x):
    return (x - 2) * _u_grad_first_order(x) - _eps * _u_grad_second_order(x)


def Partial_D(u, X):
    u_x = torch.autograd.grad(
        u, X, torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    return u_x, u_xx


def pde(X, u_x, u_xx):
    forcing = _f_right(X).data
    f_u = (X - 2) * u_x - _eps * u_xx - forcing
    return f_u


def loss(data, x_lb, x_ub, col_weights, u_weights):
    u_pred = model(data)
    u_x, u_xx = Partial_D(u_pred, data)
    f_u_pred = pde(data, u_x, u_xx)
    f_loss = torch.square(col_weights.cuda() * f_u_pred).mean()
    u_lb_pred, u_x_ub_pred = model(x_lb), model(x_ub)
    b_loss = (
        torch.square(torch.concat([u_lb_pred, u_x_ub_pred], 0) - 0) * u_weights.cuda()
    ).mean()
    b_loss_weight = max(10, min(100 / (epoch + 1), 100))
    loss_value = f_loss + b_loss * b_loss_weight
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


def train_StageI(
    data, x_lb, x_ub, col_weights, u_weights, Partial_D, pde, loss, inner_epoch=100
):
    model.train()
    for k in range(inner_epoch):
        loss_value = loss(data, x_lb, x_ub, col_weights, u_weights)
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
    data, x_lb, x_ub, col_weights, u_weights, Partial_D, pde, loss, inner_epoch=100
):
    model.train()
    if (epoch + 1) <= 300:
        bro = epoch
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
        u_x, u_xx = Partial_D(u_pred, data)
        f_u_pred = pde(data, u_x, u_xx)

        if (bro) <= 300:
            loss_f, _ = torch.topk(
                f_u_pred, int(select_rate(bro) * len(f_u_pred)), dim=0, largest=False
            )
        else:
            loss_f = f_u_pred

        u_b_pred = model(x_lb)
        mse_b_u = (torch.square((u_b_pred - x_ub))).mean()

        mse_f_u = (torch.square(loss_f)).mean()

        b_weights = max(10, min(100 / (epoch + 1), 100))
        loss_value = mse_f_u + mse_b_u * b_weights
        optimizer.zero_grad()
        loss_value.sum().backward()
        optimizer.step()
    return model, loss_value, col_weights, u_weights


for epoch in range(epochs):
    start = time.time()
    model, loss_value, col_weights, u_weights = train_StageI(
        data,
        x_lb,
        x_ub,
        col_weights,
        u_weights,
        Partial_D,
        pde,
        loss,
        inner_epoch=10,
    )
    model, loss_value, col_weights, u_weights = train_StageII(
        data,
        x_lb,
        x_ub,
        col_weights,
        u_weights,
        Partial_D,
        pde,
        loss,
        inner_epoch=1,
    )
    scheduler.step()
    p = np.linspace(0, 1, 1001)
    p = p[(p != 0) & (p != 1)]
    test_data = torch.tensor(
        p.reshape(-1, 1), dtype=torch.float32, requires_grad=True
    ).cuda()
    u_test = model(test_data)
    u_t_tst, u_xx_tst = Partial_D(u_test, test_data)
    f_u_pred_test = pde(test_data, u_t_tst, u_xx_tst)
    u_b_pred_test = model(torch.tensor([[0], [1]], dtype=torch.float32).cuda())

    mse_f_u = torch.square(f_u_pred_test).mean()
    mse_b_u = torch.square((u_b_pred_test)).mean()
    Valid = mse_f_u + mse_b_u

    if Valid < best:
        best = Valid
        torch.save(model.state_dict(), "./Best_Shockwave.pt")

    if (epoch + 1) % 1 == 0:
        time_p = time.time() - start
        interval += time_p
        text = f"Epoch:[{epoch+1}/{epochs}],Loss:{loss_value}, Valid: {Valid},Time: {time_p}s,Total Time:{interval}s."
        logger.info(text)
        start = time.time()

    if (epoch + 1) % 100 == 0:
        # 绘制图像，保存模型
        # 在任何时候，都不应出现在真实数据，这里的真实数据主要是用于画图
        datasets = np.linspace(0, 1, 1001)
        Truth = np.cos(math.pi * datasets / 2) * (1 - np.exp(-2 * datasets / _eps))
        preds = (
            model(
                torch.tensor(data=datasets.reshape(-1, 1), dtype=torch.float32).cuda()
            )
            .cpu()
            .detach()
            .numpy()
            .ravel()
        )
        figs, ax = plt.subplots(1, 2, figsize=(12, 8))

        ax[0].plot(datasets, Truth, "r-", label="Truth")
        ax[0].plot(datasets, preds, "b--", label="Pred")
        ax[0].set_title("Truth")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("U-Value")
        ax[0].grid()
        indes = np.argsort(X_f.ravel())
        col_ = col_weights.cpu().detach().numpy().ravel()[indes]
        np.save(f"./weights/{epoch+1}_col_weights.npy", col_)

        # 创建权重散点图
        xxx = np.linspace(0, len(col_), len(col_))
        sizes = (col_ - min(col_)) / (max(col_) - min(col_)) * 100 + 10  # 调整点大小
        colors = col_  # 颜色根据权重变化

        ax[1] = figs.add_subplot(122)
        scatter = ax[1].scatter(xxx, col_, s=sizes, c=colors, cmap="jet", alpha=0.75)

        # 添加颜色条，显示权重大小
        cbar = plt.colorbar(scatter, ax=ax[1])
        cbar.set_label("Weight Magnitude")

        # 添加标题和标签
        ax[1].set_title("Weight Scatter Plot")
        ax[1].set_xlabel("Index")
        ax[1].set_ylabel("Weight Value")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        torch.save(model.state_dict(), f"./model_save/Model_{epoch+1}.pt")
        plt.savefig(f"./figs/fig_epoch={epoch+1}.png")
        plt.close()
