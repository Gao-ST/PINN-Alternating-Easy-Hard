import torch
from torch import nn
import numpy as np
import os
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_inline import backend_inline
import time


import logging

logger = logging.getLogger("Helmholtz4pi")
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

folder_path1 = "./model_save"
folder_path2 = "./figs"
folder_path3 = "./weights"


def create_filedir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Created", folder_path)
    else:
        print("Existed", folder_path)


create_filedir(folder_path1)
create_filedir(folder_path2)
create_filedir(folder_path3)
# --------------- Dataset ----------------------
import math

k0 = 4 * math.pi
# Dataset
N_f = 1200
N = 234
lb = np.array([0, 0])
ub = np.array([1, 1])


a = np.linspace(0, 1, N)
b = np.linspace(0, 1, N)
A, B = np.meshgrid(a, b, indexing="ij")
data = np.concatenate([A.flatten()[:, None], B.flatten()[:, None]], 1)  # 缺少 t = 1


x = np.linspace(0, 1, N_f)
t = np.linspace(0, 1, N_f)
x0 = np.concatenate([x.reshape(-1, 1), np.zeros(shape=(len(x), 1))], 1)
x1 = np.concatenate([x.reshape(-1, 1), np.ones(shape=(len(x), 1))], 1)

xlft = np.concatenate([np.zeros(shape=(len(t), 1)), t.reshape(-1, 1)], 1)
xr = np.concatenate([np.ones(shape=(len(t), 1)), t.reshape(-1, 1)], 1)

xb = np.concatenate([x0, x1, xlft, xr], 0)
ub = np.zeros((len(xb), 1))
xb = torch.tensor(xb, dtype=torch.float32, requires_grad=True).cuda()
ub = torch.tensor(ub, dtype=torch.float32, requires_grad=True).cuda()
data = torch.tensor(data, dtype=torch.float32, requires_grad=True).cuda()

col_weights = torch.nn.Parameter(torch.rand([data.shape[0], 1]), requires_grad=True)
u_weights = torch.nn.Parameter(
    torch.ones(ub.shape) * torch.tensor([int(1e4)]), requires_grad=True
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


# --------------- Equation Config ----------------------
def _u(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


def Partial_D(u, x):
    grad = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad[:, 0:1]
    u_t = grad[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][
        :, 0:1
    ]
    u_tt = torch.autograd.grad(u_t, x, torch.ones_like(u_t), create_graph=True)[0][
        :, 1:2
    ]
    return u_xx, u_tt


def pde(x, u_tt, u_xx, k0=k0):
    u = model(x)
    return (
        k0**2 * torch.sin(k0 * x[:, 1:2]) * torch.sin(k0 * x[:, 0:1])
        + u_xx
        + u_tt
        + k0**2 * u
    )


from torch.utils.data import TensorDataset, DataLoader

target = torch.zeros((len(data), 1))
dataset = TensorDataset(data, target)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def loss(x_f, xb, ub, col_weights, u_weights):
    u_pred = model(x_f)
    u_tt, u_xx = Partial_D(u_pred, x_f)
    f_u_pred = pde(x_f, u_tt, u_xx)
    u_b_pred = model(xb)
    mse_b_u = (torch.square((u_b_pred - ub)) * (torch.pow(u_weights.cuda(), 2))).mean()
    mse_f_u = (torch.square(f_u_pred) * (torch.pow(col_weights.cuda(), 2))).mean()
    b_weights = max(10, min(100 / (epoch + 1), 100))
    return mse_b_u * b_weights + mse_f_u


epochs = 100000
lr = 0.001
col_lr = 0.001
u_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_coll = torch.optim.Adam([col_weights], lr=lr)
optimizer_u = torch.optim.Adam([u_weights], lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2100, eta_min=0, last_epoch=-1, verbose=False
)
best = 900000
interval = 0


def train_StageI(
    data, xb, ub, col_weights, u_weights, Partial_D, pde, loss, epoch, inner_epoch=100
):
    model.train()
    for k in range(inner_epoch):
        loss_value = loss(data, xb, ub, col_weights, u_weights)
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
    data, xb, ub, col_weights, u_weights, Partial_D, pde, loss, epoch, inner_epoch=1000
):
    model.train()

    if (epoch + 1) <= 300:
        bro = epoch + 1
    else:
        bro = (epoch + 1) % 300

    def select_rate(bro):
        rate = 0.5 + 0.99 * (bro + 1) / 300
        if rate > 0.99:
            return 0.99
        else:
            return rate

    for k in range(inner_epoch):
        u_pred = model(data)
        u_xx, u_tt = Partial_D(u_pred, data)
        f_u_pred = pde(data, u_xx, u_tt)

        if bro <= 150:
            _, indexx = torch.topk(
                f_u_pred * col_weights.cuda(),
                int(select_rate(bro) * len(f_u_pred)),
                dim=0,
                largest=False,
            )
            loss_f = f_u_pred[indexx].reshape(-1, 1)
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
            data,
            xb,
            ub,
            col_weights,
            u_weights,
            Partial_D,
            pde,
            loss,
            epoch,
            inner_epoch=1,
        )
    scheduler.step()
    p = np.linspace(0, 1, 201)
    q = np.linspace(0, 1, 201)
    P, Q = np.meshgrid(p, q, indexing="ij")
    datasets = np.concatenate([P.flatten()[:, None], Q.flatten()[:, None]], 1)
    pq0 = np.concatenate([p.reshape(-1, 1), np.zeros(shape=(len(p), 1))], 1)
    pq1 = np.concatenate([p.reshape(-1, 1), np.ones(shape=(len(p), 1))], 1)
    pqlft = np.concatenate([-np.ones(shape=(len(q), 1)), q.reshape(-1, 1)], 1)
    pqr = np.concatenate([np.ones(shape=(len(q), 1)), q.reshape(-1, 1)], 1)

    pqb = np.concatenate([pq0, pq1, pqlft, pqr], 0)
    upqb = np.concatenate(np.zeros((len(pqb), 1)))

    test_xb = torch.tensor(pqb, dtype=torch.float32, requires_grad=True).cuda()
    test_ub = torch.tensor(upqb, dtype=torch.float32, requires_grad=True).cuda()
    test_data = torch.tensor(datasets, dtype=torch.float32, requires_grad=True).cuda()
    u_test = model(test_data)
    u_b_pred_test = model(test_xb)
    u_tt_tst, u_xx_tst = Partial_D(u_test, test_data)

    f_u_pred_test = pde(test_data, u_tt_tst, u_xx_tst)

    mse_f_u = torch.square(f_u_pred_test).mean()
    mse_b_u = torch.square((u_b_pred_test - test_ub)).mean()
    Valid = mse_f_u + mse_b_u

    predd = u_test.cpu().detach().numpy().ravel()
    TT = (np.sin(k0 * P) * np.sin(k0 * Q)).ravel()
    rel2 = np.linalg.norm(predd - TT, 2) / np.linalg.norm(TT, 2)
    np.save("col_weights.npy", col_weights.cpu().detach().numpy().reshape(N, N))
    if rel2 < best:
        best = rel2
        torch.save(model.state_dict(), "./Best.pt")
    if (epoch + 1) % 1 == 0:
        time_p = time.time() - start
        interval += time_p
        text = f"Epoch:[{epoch+1}/{epochs}],Loss:{loss_value}, Valid: {Valid},Time: {time_p}s,Total Time:{interval}s."
        logger.info(text)
        start = time.time()

    if (epoch + 1) % 1000 == 0:
        p = np.linspace(0, 1, 101)
        q = np.linspace(0, 1, 101)
        P, Q = np.meshgrid(p, q, indexing="ij")
        datasets = np.concatenate([P.flatten()[:, None], Q.flatten()[:, None]], 1)
        Truth = np.sin(k0 * P) * np.sin(k0 * Q)

        preds = (
            model(torch.tensor(data=datasets, dtype=torch.float32).cuda())
            .cpu()
            .detach()
            .numpy()
            .reshape(101, 101)
        )
        extent = [0, 1, 0, 1]
        figs, ax = plt.subplots(2, 2, figsize=(18, 18))
        ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

        h1 = ax1.imshow(Truth, cmap="jet", aspect="auto", extent=extent)
        ax1.set_title("Truth")
        ax1.set_xlabel("T")
        ax1.set_ylabel("X")
        figs.colorbar(h1, ax=ax1, shrink=0.5, aspect=30, pad=0.05)

        h2 = ax2.imshow(preds, cmap="jet", aspect="auto", extent=extent)
        ax2.set_title("Preds")
        ax2.set_xlabel("T")
        ax2.set_ylabel("X")
        figs.colorbar(h2, ax=ax2, shrink=0.5, aspect=30, pad=0.05)

        h3 = ax3.imshow(np.abs(Truth - preds), cmap="jet", aspect="auto", extent=extent)
        ax3.set_title("Error")
        ax3.set_xlabel("T")
        ax3.set_ylabel("X")
        figs.colorbar(h3, ax=ax3, shrink=0.5, aspect=30, pad=0.05)

        h4 = ax4.imshow(
            col_weights.cpu().detach().numpy().reshape(N, N),
            cmap="jet",
            aspect="auto",
            extent=extent,
        )
        ax4.set_title("Weights")
        ax4.set_xlabel("T")
        ax4.set_ylabel("X")
        figs.colorbar(h4, ax=ax4, shrink=0.5, aspect=30, pad=0.05)

        torch.save(model.state_dict(), f"./model_save/Model_{epoch+1}.pt")
        plt.savefig(f"./figs/fig_epoch={epoch+1}.png")
        plt.close()
