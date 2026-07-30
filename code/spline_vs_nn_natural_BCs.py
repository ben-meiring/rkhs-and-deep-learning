from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = Path(__file__).resolve().parent

seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 1. Natural cubic smoothing spline (BCs on f'', f''')
# ============================================================

def G_kernel_torch(x, xi):
    # Green-like kernel piece: |x - xi|^3 / 12
    return torch.abs(x - xi) ** 3 / 12.0


def solve_nat_spline_torch(x, y, lam):
    """
    Natural cubic smoothing spline:
        minimize sum_i (y_i - f(x_i))^2 + lam * ∫ (f'')^2
    with
        f(x) = a0 + a1 * x + sum_j c_j * |x - x_j|^3 / 12
    and natural BCs:
        f''(a)=f''(b)=f'''(a)=f'''(b)=0
    implemented via:
        sum_j c_j = 0,  sum_j c_j x_j = 0
    """
    x = x.flatten().to(device)
    y = y.flatten().to(device)
    N = x.size(0)
    dtype = x.dtype

    K = torch.empty((N, N), dtype=dtype, device=device)
    for i in range(N):
        for j in range(N):
            K[i, j] = G_kernel_torch(x[i], x[j])

    X = torch.stack([torch.ones_like(x), x], dim=1)

    A11 = K + lam * torch.eye(N, dtype=dtype, device=device)
    A12 = X
    A21 = X.t()
    A22 = torch.zeros(2, 2, dtype=dtype, device=device)

    top = torch.cat([A11, A12], dim=1)
    bottom = torch.cat([A21, A22], dim=1)
    A = torch.cat([top, bottom], dim=0)

    rhs = torch.cat([y, torch.zeros(2, dtype=dtype, device=device)], dim=0)
    sol = torch.linalg.solve(A, rhs)

    c = sol[:N]
    d = sol[N:]
    a0, a1 = d[0], d[1]
    return c, a0, a1, K


def f_spline_torch(x_eval, x_data, c, a0, a1):
    """
    Evaluate f(x) = a0 + a1 * x + sum_j c_j * |x - x_j|^3 / 12
    at x_eval.
    """
    x_eval = x_eval.flatten().to(device)
    x_data = x_data.flatten().to(device)
    G_sum = torch.zeros_like(x_eval)
    N = x_data.size(0)
    for j in range(N):
        G_sum += c[j] * G_kernel_torch(x_eval, x_data[j])
    return a0 + a1 * x_eval + G_sum


# ============================================================
# 2. Neural network model
# ============================================================

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def eval_nn_and_derivs(x_scalar, model):
    x = torch.tensor([[x_scalar]], device=device, requires_grad=True, dtype=torch.float64)
    f = model(x)
    df_dx = torch.autograd.grad(
        outputs=f, inputs=x,
        grad_outputs=torch.ones_like(f),
        create_graph=True, retain_graph=True
    )[0]
    d2f_dx2 = torch.autograd.grad(
        outputs=df_dx, inputs=x,
        grad_outputs=torch.ones_like(df_dx),
        create_graph=True, retain_graph=True
    )[0]
    d3f_dx3 = torch.autograd.grad(
        outputs=d2f_dx2, inputs=x,
        grad_outputs=torch.ones_like(d2f_dx2),
        create_graph=True, retain_graph=True
    )[0]
    return f.item(), df_dx.item(), d2f_dx2.item(), d3f_dx3.item()


# ============================================================
# 3. Data and NN training
# ============================================================

N_train = 30
x_train = torch.linspace(0.0, 1.0, N_train + 2, device=device, dtype=torch.float64)[1:-1].unsqueeze(1)
noise_std = 0.25
y_train = torch.sin(2.0 * torch.pi * x_train) + noise_std * torch.randn_like(x_train)

model = SimpleNet().to(device).double()
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 20000
lambda_curv = 1e-4
lambda_eff = N_train * lambda_curv

spline_linewidth = 2.8
nn_linewidth = 1.6
spline_zorder = 2
nn_zorder = 3

label_fontsize = 15
title_fontsize = 16
legend_fontsize = 12
tick_fontsize = 12

mse_loss = nn.MSELoss()

N_reg = 1000
x_reg_base = torch.linspace(0.0, 1.0, N_reg, device=device, dtype=torch.float64).unsqueeze(1)

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(x_train)
    loss_data = mse_loss(y_pred, y_train)

    x_reg = x_reg_base.clone().detach().requires_grad_(True)
    f_reg = model(x_reg)
    df_dx = torch.autograd.grad(
        outputs=f_reg, inputs=x_reg,
        grad_outputs=torch.ones_like(f_reg),
        create_graph=True, retain_graph=True
    )[0]
    d2f_dx2 = torch.autograd.grad(
        outputs=df_dx, inputs=x_reg,
        grad_outputs=torch.ones_like(df_dx),
        create_graph=True, retain_graph=True
    )[0]
    curvature_penalty = lambda_curv * (d2f_dx2 ** 2).mean()

    loss = loss_data + curvature_penalty
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch {epoch+1:4d} | "
            f"Data loss: {loss_data.item():.6e} | "
            f"Curv: {curvature_penalty.item():.6e} | "
            f"Total: {loss.item():.6e}"
        )

# ============================================================
# 4. Analytic spline solution with natural BCs
# ============================================================

x_train_flat = x_train.detach().clone().double().flatten()
y_train_flat = y_train.detach().clone().double().flatten()

c, a0, a1, K = solve_nat_spline_torch(x_train_flat, y_train_flat, lambda_eff)

f_train_spline = f_spline_torch(x_train_flat, x_train_flat, c, a0, a1)
mse_spline = torch.mean((f_train_spline - y_train_flat) ** 2)
print(f"\nAnalytic spline MSE loss: {mse_spline.item():.4e}")

x_reg_dense = torch.linspace(0.0, 1.0, 5000, device=device, dtype=torch.float64).requires_grad_(True)
f_reg_spline = f_spline_torch(x_reg_dense, x_train_flat, c, a0, a1)

df_dx_spline = torch.autograd.grad(
    outputs=f_reg_spline, inputs=x_reg_dense,
    grad_outputs=torch.ones_like(f_reg_spline),
    create_graph=True, retain_graph=True
)[0]
d2f_dx2_spline = torch.autograd.grad(
    outputs=df_dx_spline, inputs=x_reg_dense,
    grad_outputs=torch.ones_like(df_dx_spline),
    create_graph=True, retain_graph=True
)[0]

curv_spline = lambda_curv * (d2f_dx2_spline ** 2).mean()
total_loss_spline = mse_spline + curv_spline

x_reg_nn = x_reg_dense.clone().detach().requires_grad_(True).unsqueeze(1)
f_reg_nn = model(x_reg_nn)
df_dx_nn = torch.autograd.grad(
    outputs=f_reg_nn, inputs=x_reg_nn,
    grad_outputs=torch.ones_like(f_reg_nn),
    create_graph=True, retain_graph=True
)[0]
d2f_dx2_nn = torch.autograd.grad(
    outputs=df_dx_nn, inputs=x_reg_nn,
    grad_outputs=torch.ones_like(df_dx_nn),
    create_graph=True, retain_graph=True
)[0]

curv_nn = lambda_curv * (d2f_dx2_nn ** 2).mean()

model.eval()
with torch.no_grad():
    y_pred_nn_train = model(x_train)
mse_nn = torch.mean((y_pred_nn_train.flatten() - y_train_flat) ** 2)
total_loss_nn = mse_nn + curv_nn

print("\n--- Loss on Dense Grid ---")
print(f"Analytic spline:")
print(f"    MSE loss          : {mse_spline.item():.6e}")
print(f"    Curvature penalty : {curv_spline.item():.6e}")
print(f"    Total loss        : {total_loss_spline.item():.6e}")

print(f"\nNeural network:")
print(f"    MSE loss          : {mse_nn.item():.6e}")
print(f"    Curvature penalty : {curv_nn.item():.6e}")
print(f"    Total loss        : {total_loss_nn.item():.6e}")

# ============================================================
# 5. Boundary checks and plotting
# ============================================================

x0 = torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True)
x1 = torch.tensor([1.0], device=device, dtype=torch.float64, requires_grad=True)

f0_spline = f_spline_torch(x0, x_train_flat, c, a0, a1)
f1_spline = f_spline_torch(x1, x_train_flat, c, a0, a1)

df0_spline = torch.autograd.grad(f0_spline, x0, grad_outputs=torch.ones_like(f0_spline), create_graph=True)[0]
df1_spline = torch.autograd.grad(f1_spline, x1, grad_outputs=torch.ones_like(f1_spline), create_graph=True)[0]

d2f0_spline = torch.autograd.grad(df0_spline, x0, grad_outputs=torch.ones_like(df0_spline), create_graph=True)[0]
d2f1_spline = torch.autograd.grad(df1_spline, x1, grad_outputs=torch.ones_like(df1_spline), create_graph=True)[0]

d3f0_spline = torch.autograd.grad(d2f0_spline, x0, grad_outputs=torch.ones_like(d2f0_spline))[0]
d3f1_spline = torch.autograd.grad(d2f1_spline, x1, grad_outputs=torch.ones_like(d2f1_spline))[0]

print("\nAnalytic spline at boundaries:")
print(f"  f(0)   = {f0_spline.item():.6f},   f(1)   = {f1_spline.item():.6f}")
print(f"  f'(0)  = {df0_spline.item():.6f},  f'(1)  = {df1_spline.item():.6f}")
print(f"  f''(0) = {d2f0_spline.item():.6f}, f''(1) = {d2f1_spline.item():.6f}")
print(f"  f'''(0)= {d3f0_spline.item():.6f}, f'''(1)= {d3f1_spline.item():.6f}")

f0_nn, df0_nn, d2f0_nn, d3f0_nn = eval_nn_and_derivs(0.0, model)
f1_nn, df1_nn, d2f1_nn, d3f1_nn = eval_nn_and_derivs(1.0, model)

print("\nNeural network at boundaries:")
print(f"  f(0)   = {f0_nn:.6f},   f(1)   = {f1_nn:.6f}")
print(f"  f'(0)  = {df0_nn:.6f},  f'(1)  = {df1_nn:.6f}")
print(f"  f''(0) = {d2f0_nn:.6f}, f''(1) = {d2f1_nn:.6f}")
print(f"  f'''(0) = {d3f0_nn:.6f}, f'''(1) = {d3f1_nn:.6f}")

x_test = torch.linspace(0.0, 1.0, 200, device=device, dtype=torch.float64).unsqueeze(1)
y_test = torch.sin(2.0 * torch.pi * x_test)

with torch.no_grad():
    y_pred_nn_test = model(x_test)

x_test_np = x_test.cpu().numpy()
y_test_np = y_test.cpu().numpy()
y_pred_nn_np = y_pred_nn_test.cpu().numpy()

x_test_flat = x_test.flatten()
f_vals = f_spline_torch(x_test_flat, x_train_flat, c, a0, a1).view_as(x_test)
f_vals_np = f_vals.detach().cpu().numpy()

fig1 = plt.figure(figsize=(8, 5))
plt.plot(x_test_np, y_test_np, label=r"Target: $\sin(2\pi x)$", color="black", linewidth=2, zorder=1)
plt.plot(
    x_test_np,
    f_vals_np,
    label="Analytic natural spline",
    color="green",
    linewidth=spline_linewidth,
    zorder=spline_zorder,
)
plt.plot(
    x_test_np,
    y_pred_nn_np,
    label="Neural network",
    color="red",
    linewidth=nn_linewidth,
    zorder=nn_zorder,
)
plt.scatter(
    x_train.cpu().numpy(),
    y_train.cpu().numpy(),
    color="blue",
    s=10,
    label="Training points",
    zorder=4,
)
plt.legend(fontsize=legend_fontsize)
plt.xlabel(r"$x$", fontsize=label_fontsize)
plt.ylabel(r"$f(x)$", fontsize=label_fontsize)
plt.title(r"Neural Network vs. Natural Cubic Smoothing Spline", fontsize=title_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)
fig1.savefig(save_dir / "nn_vs_natural_spline.png", dpi=300, bbox_inches="tight")
plt.show()

x_plot = torch.linspace(0.0, 1.0, 500, device=device, dtype=torch.float64).requires_grad_(True)
f_plot_spline = f_spline_torch(x_plot, x_train_flat, c, a0, a1)

df_plot_spline = torch.autograd.grad(
    f_plot_spline, x_plot,
    grad_outputs=torch.ones_like(f_plot_spline),
    create_graph=True, retain_graph=True
)[0]
d2f_plot_spline = torch.autograd.grad(
    df_plot_spline, x_plot,
    grad_outputs=torch.ones_like(df_plot_spline),
    create_graph=True, retain_graph=True
)[0]
d3f_plot_spline = torch.autograd.grad(
    d2f_plot_spline, x_plot,
    grad_outputs=torch.ones_like(d2f_plot_spline),
    create_graph=True, retain_graph=True
)[0]

x_plot_nn = x_plot.clone().detach().requires_grad_(True).unsqueeze(1)
f_plot_nn = model(x_plot_nn)
df_plot_nn = torch.autograd.grad(
    f_plot_nn, x_plot_nn,
    grad_outputs=torch.ones_like(f_plot_nn),
    create_graph=True, retain_graph=True
)[0].squeeze(1)
d2f_plot_nn = torch.autograd.grad(
    df_plot_nn, x_plot_nn,
    grad_outputs=torch.ones_like(df_plot_nn),
    create_graph=True, retain_graph=True
)[0].squeeze(1)
d3f_plot_nn = torch.autograd.grad(
    d2f_plot_nn, x_plot_nn,
    grad_outputs=torch.ones_like(d2f_plot_nn),
    create_graph=True, retain_graph=True
)[0].squeeze(1)

x_plot_np = x_plot.detach().cpu().numpy()
f_plot_spline_np = f_plot_spline.detach().cpu().numpy()
df_plot_spline_np = df_plot_spline.detach().cpu().numpy()
d2f_plot_spline_np = d2f_plot_spline.detach().cpu().numpy()
d3f_plot_spline_np = d3f_plot_spline.detach().cpu().numpy()

f_plot_nn_np = f_plot_nn.detach().cpu().numpy()
df_plot_nn_np = df_plot_nn.detach().cpu().numpy()
d2f_plot_nn_np = d2f_plot_nn.detach().cpu().numpy()
d3f_plot_nn_np = d3f_plot_nn.detach().cpu().numpy()

fig2 = plt.figure(figsize=(10, 12))

plt.subplot(4, 1, 1)
plt.plot(x_plot_np, f_plot_spline_np, label="Analytic spline", color="green", linewidth=spline_linewidth, zorder=spline_zorder)
plt.plot(x_plot_np, f_plot_nn_np, label="Neural network", color="red", linestyle="--", linewidth=nn_linewidth, zorder=nn_zorder)
plt.ylabel(r"$f(x)$", fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.title(r"Function and Derivatives", fontsize=title_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)

plt.subplot(4, 1, 2)
plt.plot(x_plot_np, df_plot_spline_np, label="Analytic spline", color="green", linewidth=spline_linewidth, zorder=spline_zorder)
plt.plot(x_plot_np, df_plot_nn_np, label="Neural network", color="red", linestyle="--", linewidth=nn_linewidth, zorder=nn_zorder)
plt.ylabel(r"$f'(x)$", fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)

plt.subplot(4, 1, 3)
plt.plot(x_plot_np, d2f_plot_spline_np, label="Analytic spline", color="green", linewidth=spline_linewidth, zorder=spline_zorder)
plt.plot(x_plot_np, d2f_plot_nn_np, label="Neural network", color="red", linestyle="--", linewidth=nn_linewidth, zorder=nn_zorder)
plt.ylabel(r"$f''(x)$", fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)

plt.subplot(4, 1, 4)
plt.plot(x_plot_np, d3f_plot_spline_np, label="Analytic spline", color="green", linewidth=spline_linewidth, zorder=spline_zorder)
plt.plot(x_plot_np, d3f_plot_nn_np, label="Neural network", color="red", linestyle="--", linewidth=nn_linewidth, zorder=nn_zorder)
plt.xlabel(r"$x$", fontsize=label_fontsize)
plt.ylabel(r"$f'''(x)$", fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)

plt.tight_layout()
fig2.savefig(save_dir / "spline_vs_nn_derivatives.png", dpi=300, bbox_inches="tight")
plt.show()
