import csv
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row_dict):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_dict)


# ============================================================
# 1D PDE evaluation (Allen-Cahn, Burgers, KdV)
# ============================================================

def evaluate_l2_1d(params, model, t_star, x_star, u_ref):
    TT, XX = jnp.meshgrid(t_star, x_star, indexing='ij')
    eval_grid = jnp.stack([TT.ravel(), XX.ravel()], axis=-1)

    batch_size = 4096
    u_pred_list = []
    for start in range(0, eval_grid.shape[0], batch_size):
        end = min(start + batch_size, eval_grid.shape[0])
        u_batch = model.apply(params, eval_grid[start:end])[:, 0]
        u_pred_list.append(u_batch)

    u_pred = jnp.concatenate(u_pred_list).reshape(len(t_star), len(x_star))
    l2 = float(jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref))
    return l2, u_pred


def plot_1d(t_star, x_star, u_ref, u_pred, l2, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title} (L2={l2:.4e})', fontsize=14)

    im0 = axes[0].pcolormesh(t_star, x_star, u_ref.T, cmap='jet', shading='auto')
    axes[0].set_title('Reference'); axes[0].set_xlabel('t'); axes[0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(t_star, x_star, np.array(u_pred).T, cmap='jet', shading='auto')
    axes[1].set_title('Predicted'); axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(np.array(u_pred) - u_ref)
    im2 = axes[2].pcolormesh(t_star, x_star, err.T, cmap='hot', shading='auto')
    axes[2].set_title('Error'); axes[2].set_xlabel('t'); axes[2].set_ylabel('x')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.show()


# ============================================================
# 2D NS evaluation (lid-driven cavity)
# ============================================================

def evaluate_ldc(params, model, x_ref, y_ref, u_ref, v_ref, t_eval):
    nx, ny = len(x_ref), len(y_ref)
    XX, YY = jnp.meshgrid(x_ref, y_ref, indexing='ij')
    coords = jnp.stack([
        jnp.full(nx * ny, t_eval),
        XX.ravel(), YY.ravel()
    ], axis=-1)

    batch_size = 4096
    out_list = []
    for start in range(0, coords.shape[0], batch_size):
        end = min(start + batch_size, coords.shape[0])
        out_list.append(model.apply(params, coords[start:end]))

    out = jnp.concatenate(out_list, axis=0)
    u_pred = out[:, 0].reshape(nx, ny)
    v_pred = out[:, 1].reshape(nx, ny)

    l2_u = float(jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref))
    l2_v = float(jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref))
    l2_total = float(
        jnp.sqrt(jnp.sum((u_pred - u_ref)**2) + jnp.sum((v_pred - v_ref)**2))
        / jnp.sqrt(jnp.sum(u_ref**2) + jnp.sum(v_ref**2))
    )
    return l2_u, l2_v, l2_total, u_pred, v_pred


def plot_ldc(x_ref, y_ref, u_ref, v_ref, u_pred, v_pred, l2_total, Re, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Lid-Driven Cavity Re={Re:.0f}  (L2={l2_total:.4e})', fontsize=14)

    u_pred_np, v_pred_np = np.array(u_pred), np.array(v_pred)

    for ax, data, title in [
        (axes[0, 0], u_ref, 'u Reference'),
        (axes[0, 1], u_pred_np, 'u Predicted'),
        (axes[0, 2], np.abs(u_pred_np - u_ref), 'u Error'),
        (axes[1, 0], v_ref, 'v Reference'),
        (axes[1, 1], v_pred_np, 'v Predicted'),
        (axes[1, 2], np.abs(v_pred_np - v_ref), 'v Error'),
    ]:
        cmap = 'hot' if 'Error' in title else 'jet'
        im = ax.pcolormesh(x_ref, y_ref, data.T, cmap=cmap, shading='auto')
        ax.set_title(title); ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.show()


def plot_ldc_centerline(x_ref, y_ref, u_ref, v_ref, u_pred, v_pred, Re, save_path):
    u_pred_np, v_pred_np = np.array(u_pred), np.array(v_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Centerline Profiles — Re={Re:.0f}', fontsize=14)

    ix_mid = np.argmin(np.abs(x_ref - 0.5))
    axes[0].plot(u_ref[ix_mid, :], y_ref, 'k-', lw=2, label='Reference')
    axes[0].plot(u_pred_np[ix_mid, :], y_ref, 'r--', lw=2, label='PINN')
    axes[0].set_xlabel('u'); axes[0].set_ylabel('y')
    axes[0].set_title('u at x=0.5'); axes[0].legend(); axes[0].grid(True)

    iy_mid = np.argmin(np.abs(y_ref - 0.5))
    axes[1].plot(x_ref, v_ref[:, iy_mid], 'k-', lw=2, label='Reference')
    axes[1].plot(x_ref, v_pred_np[:, iy_mid], 'r--', lw=2, label='PINN')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('v')
    axes[1].set_title('v at y=0.5'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.show()


# ============================================================
# 2D NS animation (channel flow / cavity)
# ============================================================

def predict_2d_field(params, model, t_val, x_range, y_range, nx=100, ny=50):
    x_lin = jnp.linspace(x_range[0], x_range[1], nx)
    y_lin = jnp.linspace(y_range[0], y_range[1], ny)
    XX, YY = jnp.meshgrid(x_lin, y_lin, indexing='ij')

    coords = jnp.stack([
        jnp.full(nx * ny, t_val), XX.ravel(), YY.ravel()
    ], axis=-1)

    batch_size = 2048
    out_list = []
    for start in range(0, coords.shape[0], batch_size):
        end = min(start + batch_size, coords.shape[0])
        out_list.append(model.apply(params, coords[start:end]))

    out = jnp.concatenate(out_list, axis=0)
    u = out[:, 0].reshape(nx, ny)
    v = out[:, 1].reshape(nx, ny)
    p = out[:, 2].reshape(nx, ny)
    return x_lin, y_lin, u, v, p


def animate_2d(params, model, x_range, y_range, t_max,
               n_frames=50, nx=100, ny=50, interval=150):

    t_vals = jnp.linspace(0, t_max, n_frames)

    print("Computing frames...")
    frames = []
    for i, t_val in enumerate(t_vals):
        x_lin, y_lin, u, v, p = predict_2d_field(
            params, model, float(t_val), x_range, y_range, nx, ny)
        frames.append((float(t_val), np.array(u), np.array(v), np.array(p)))
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    u_min = min(f[1].min() for f in frames)
    u_max = max(f[1].max() for f in frames)
    v_abs = max(max(abs(f[2].min()), abs(f[2].max())) for f in frames)
    p_abs = max(max(abs(f[3].min()), abs(f[3].max())) for f in frames)
    v_abs = max(v_abs, 1e-6)
    p_abs = max(p_abs, 1e-6)

    x_np, y_np = np.array(x_lin), np.array(y_lin)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35)

    im0 = axes[0].pcolormesh(x_np, y_np, frames[0][1].T,
                              cmap='jet', shading='auto', vmin=u_min, vmax=u_max)
    axes[0].set_ylabel('y'); axes[0].set_title('u')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x_np, y_np, frames[0][2].T,
                              cmap='RdBu_r', shading='auto', vmin=-v_abs, vmax=v_abs)
    axes[1].set_ylabel('y'); axes[1].set_title('v')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(x_np, y_np, frames[0][3].T,
                              cmap='coolwarm', shading='auto', vmin=-p_abs, vmax=p_abs)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_title('p')
    plt.colorbar(im2, ax=axes[2])

    title = fig.suptitle(f't = {frames[0][0]:.2f}', fontsize=14)

    def update(frame_idx):
        t_val, u, v, p = frames[frame_idx]
        im0.set_array(u.T.ravel())
        im1.set_array(v.T.ravel())
        im2.set_array(p.T.ravel())
        title.set_text(f't = {t_val:.2f}')
        return im0, im1, im2, title

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    return ani
