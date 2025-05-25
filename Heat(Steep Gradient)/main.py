import torch
from torch import nn
import numpy as np
import os
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_inline import backend_inline
import time

# Logging
import logging

logger = logging.getLogger("(10,1)-0.11")
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
# --------------- MKDIR ----------------------
folder_path1 = "./model_save"
folder_path2 = "./figs"
folder_path3 = "./weights"


def create_filedir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder is Created", folder_path)
    else:
        print("Folder has Existed", folder_path)


create_filedir(folder_path1)
create_filedir(folder_path2)
create_filedir(folder_path3)
# --------------- Dataset ----------------------
eps = 0.11
N_f = 1200
N = 234
lb = np.array([-1, 0])
ub = np.array([1, 1])


a = np.linspace(-1, 1, N)
b = np.linspace(0, 1, N)
a = a[(a != 1) & (a != -1)]
b = b[b != 0]
A, B = np.meshgrid(a, b, indexing="ij")
data = np.concatenate([A.flatten()[:, None], B.flatten()[:, None]], 1)  # 缺少 t = 1


x = np.linspace(-1, 1, N_f)
t = np.linspace(0, 1, N_f)
x = x[(x != 1) & (x != -1)]
t = t[t != 0]
x0 = np.concatenate([x.reshape(-1, 1), np.zeros(shape=(len(x), 1))], 1)
xlft = np.concatenate([-np.ones(shape=(len(t), 1)), t.reshape(-1, 1)], 1)
xr = np.concatenate([np.ones(shape=(len(t), 1)), t.reshape(-1, 1)], 1)

xb = np.concatenate([x0, xlft, xr], 0)
ub = (1 - xb[:, 0:1] ** 2) * np.exp(1 / ((2 * xb[:, 1:2] - 1) ** 2 + eps))
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
    return (1 - x[:, 0:1] ** 2) * torch.exp(1 / ((2 * x[:, 1:2] - 1) ** 2 + eps))


def Partial_D(u, x):
    grad = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x = grad[:, 0:1]
    u_t = grad[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][
        :, 0:1
    ]
    return u_t, u_xx


def pde(x, u_t, u_xx, eps=eps):
    U = _u(x)
    u1, u2 = Partial_D(U, x)  # souece f
    return u_t - u_xx - u1 + u2


from torch.utils.data import TensorDataset, DataLoader

target = torch.zeros((len(data), 1))
dataset = TensorDataset(data, target)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def loss(x_f, xb, ub, col_weights, u_weights):
    u_pred = model(x_f)
    u_t, u_xx = Partial_D(u_pred, x_f)
    f_u_pred = pde(x_f, u_t, u_xx)
    u_b_pred = model(xb)
    mse_b_u = (torch.square((u_b_pred - ub)) * (torch.pow(u_weights.cuda(), 2))).mean()
    mse_f_u = (torch.square(f_u_pred) * (torch.pow(col_weights.cuda(), 2))).mean()
    return mse_b_u + mse_f_u


n = 300
epochs = 100000
lr = 0.001
col_lr = 0.001
u_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_coll = torch.optim.Adam([col_weights], lr=col_lr)
optimizer_u = torch.optim.Adam([u_weights], lr=u_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=2100, eta_min=0, last_epoch=-1, verbose=False
)
best = 2
interval = 0


def train_StageI(
    data, xb, ub, col_weights, u_weights, Partial_D, pde, loss, inner_epoch=100
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
    data, xb, ub, col_weights, u_weights, Partial_D, pde, epoch, inner_epoch=1000
):
    model.train()
    if (epoch + 1) <= n:
        bro = epoch + 1
    else:
        bro = (epoch + 1) % n

    def select_rate(bro):
        rate = 0.5 + 0.99 * (bro) / n
        if rate > 0.99:
            return 0.99
        else:
            return rate

    for k in range(inner_epoch):
        u_pred = model(data)
        u_t, u_xx = Partial_D(u_pred, data)
        f_u_pred = pde(data, u_t, u_xx)

        if (bro + 1) <= n / 2:
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


for epoch in range(epochs):
    start = time.time()
    model, loss_value, col_weights, u_weights = train_StageI(
        data, xb, ub, col_weights, u_weights, Partial_D, pde, loss, inner_epoch=10
    )
    model, loss_value, col_weights, u_weights = train_StageII(
        data, xb, ub, col_weights, u_weights, Partial_D, pde, epoch, inner_epoch=1
    )
    scheduler.step()

    p = np.linspace(-1, 1, 201)
    q = np.linspace(0, 1, 101)
    R, S = np.meshgrid(p, q, indexing="ij")
    p = p[(p != 1) & (p != -1)]
    q = q[q != 0]
    P, Q = np.meshgrid(p, q, indexing="ij")
    datasets = np.concatenate([P.flatten()[:, None], Q.flatten()[:, None]], 1)
    pq0 = np.concatenate([p.reshape(-1, 1), np.zeros(shape=(len(p), 1))], 1)
    pqlft = np.concatenate([-np.ones(shape=(len(q), 1)), q.reshape(-1, 1)], 1)
    pqr = np.concatenate([np.ones(shape=(len(q), 1)), q.reshape(-1, 1)], 1)

    pqb = np.concatenate([pq0, pqlft, pqr], 0)
    upqb = np.concatenate(
        [
            (1 - p.reshape(-1, 1) ** 2) * np.exp(1 / (1 + eps)),
            np.zeros((len(pqlft), 1)),
            np.zeros((len(pqr), 1)),
        ],
        0,
    )

    test_xb = torch.tensor(pqb, dtype=torch.float32, requires_grad=True).cuda()
    test_ub = torch.tensor(upqb, dtype=torch.float32, requires_grad=True).cuda()
    test_data = torch.tensor(datasets, dtype=torch.float32, requires_grad=True).cuda()
    u_test = model(test_data)
    u_b_pred_test = model(test_xb)
    u_t_tst, u_xx_tst = Partial_D(u_test, test_data)

    f_u_pred_test = pde(test_data, u_t_tst, u_xx_tst)

    mse_f_u = torch.square(f_u_pred_test).mean()
    mse_b_u = torch.square((u_b_pred_test - test_ub)).mean()
    Valid = mse_f_u + mse_b_u

    if Valid < best:
        best = Valid
        torch.save(model.state_dict(), "./Best_Heat_Large_Gradient.pt")

    if (epoch + 1) % 1 == 0:
        time_p = time.time() - start
        interval += time_p
        text = f"Epoch:[{epoch+1}/{epochs}], Loss:{loss_value}, Valid: {Valid}, Time: {time_p}s, Total Time:{interval}s."
        logger.info(text)

    if (epoch + 1) % 1000 == 0:
        p = np.linspace(-1, 1, 201)
        q = np.linspace(0, 1, 201)
        P, Q = np.meshgrid(p, q, indexing="ij")
        datasets = np.concatenate([P.flatten()[:, None], Q.flatten()[:, None]], 1)
        Truth = (1 - P**2) * np.exp(1 / ((2 * Q - 1) ** 2 + eps))
        preds = (
            model(torch.tensor(data=datasets, dtype=torch.float32).cuda())
            .cpu()
            .detach()
            .numpy()
            .reshape(201, 201)
        )
        extent = [0, 1, -1, 1]
        figs, ax = plt.subplots(2, 2, figsize=(14, 14))
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
            col_weights.cpu().detach().numpy().ravel().reshape(N - 2, N - 1),
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
