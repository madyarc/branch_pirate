import jax
import jax.numpy as jnp
from jax import vmap, grad


# Wave equation

class WaveEquationLoss:
    def __init__(self, model, c=1.0):
        self.model = model
        self.c = c

    @staticmethod
    def ic_fn(x):
        return jnp.cos(jnp.pi * x)

    @staticmethod
    def ic_vel_fn(x):
        return jnp.zeros_like(x)

    def ic_loss(self, params, x_ic):
        B = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic], axis=-1)
        u_pred = self.model.apply(params, coords)[:, 0]
        return jnp.mean((u_pred - self.ic_fn(x_ic)) ** 2)

    def ic_vel_loss(self, params, x_ic):
        def u_t_at(x_val):
            def u_net_t(t):
                return self.model.apply(params, jnp.array([[t, x_val]]))[0, 0]
            _, ut = jax.jvp(u_net_t, (0.0,), (1.0,))
            return ut
        ut_pred = vmap(u_t_at)(x_ic)
        return jnp.mean((ut_pred - self.ic_vel_fn(x_ic)) ** 2)

    def bc_loss(self, params, t_bc):
        def bc_single(t):
            def u_at(x_val):
                return self.model.apply(params, jnp.array([[t, x_val]]))[0, 0]
            u_l,  ux_l = jax.jvp(u_at, (-1.0,), (1.0,))
            u_r,  ux_r = jax.jvp(u_at,  (1.0,), (1.0,))
            return (u_l - u_r) ** 2 + (ux_l - ux_r) ** 2
        return jnp.mean(vmap(bc_single)(t_bc))

    def res_loss(self, params, t_res, x_res):
        c = self.c
        def u_net(t, x):
            return self.model.apply(params, jnp.array([[t, x]]))[0, 0]
        def pde_single(t, x):
            _, u_tt = jax.jvp(
                lambda t_: jax.jvp(lambda t__: u_net(t__, x), (t_,), (1.0,))[1],
                (t,), (1.0,)
            )
            _, u_xx = jax.jvp(
                lambda x_: jax.jvp(lambda x__: u_net(t, x__), (x_,), (1.0,))[1],
                (x,), (1.0,)
            )
            return u_tt - c**2 * u_xx
        return jnp.mean(vmap(pde_single)(t_res, x_res) ** 2)


# Transport equation

class TransportLoss:
    def __init__(self, model, c=1.0):
        self.model = model
        self.c     = c

    @staticmethod
    def ic_fn(x):
        return jnp.cos(jnp.pi * x)

    def ic_loss(self, params, x_ic):
        B      = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic], axis=-1)
        u_pred = self.model.apply(params, coords)[:, 0]
        return jnp.mean((u_pred - self.ic_fn(x_ic)) ** 2)

    def bc_loss(self, params, t_bc):
        def bc_single(t):
            def u_at(x_val):
                return self.model.apply(params, jnp.array([[t, x_val]]))[0, 0]
            u_l, ux_l = jax.jvp(u_at, (-1.0,), (1.0,))
            u_r, ux_r = jax.jvp(u_at,  (1.0,), (1.0,))
            return (u_l - u_r) ** 2 + (ux_l - ux_r) ** 2
        return jnp.mean(vmap(bc_single)(t_bc))

    def res_loss(self, params, t_res, x_res):
        c = self.c
        def u_net(t, x):
            return self.model.apply(params, jnp.array([[t, x]]))[0, 0]
        def pde_single(t, x):
            
            u_t = jax.grad(u_net, argnums=0)(t, x)
            u_x = jax.grad(u_net, argnums=1)(t, x)
            
            return u_t + c * u_x
        return jnp.mean(vmap(pde_single)(t_res, x_res) ** 2)

# ============================================================
# Allen-Cahn: u_t = eps * u_xx + u - u^3
# ============================================================

class AllenCahnLoss:
    def __init__(self, model, eps=0.0001):
        self.model = model
        self.eps = eps

    @staticmethod
    def ic_fn(x):
        return x ** 2 * jnp.cos(jnp.pi * x)

    def ic_loss(self, params, x_ic):
        B = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic], axis=-1)
        u_pred = self.model.apply(params, coords)[:, 0]
        return jnp.mean((u_pred - self.ic_fn(x_ic)) ** 2)

    def bc_loss(self, params, t_bc):
        def bc_single(t):
            def u_at(x_val):
                return self.model.apply(params, jnp.array([[t, x_val]]))[0, 0]
            u_l = u_at(-1.0)
            u_r = u_at(1.0)
            _, ux_l = jax.jvp(u_at, (-1.0,), (1.0,))
            _, ux_r = jax.jvp(u_at, (1.0,), (1.0,))
            return (u_l - u_r) ** 2 + (ux_l - ux_r) ** 2
        return jnp.mean(vmap(bc_single)(t_bc))

    def res_loss(self, params, t_res, x_res):
        eps = self.eps
        def u_net(t, x):
            return self.model.apply(params, jnp.array([[t, x]]))[0, 0]
        def pde_single(t, x):
            u, u_t = jax.jvp(lambda t_: u_net(t_, x), (t,), (1.0,))
            _, u_xx = jax.jvp(
                lambda x_: jax.jvp(lambda x__: u_net(t, x__), (x_,), (1.0,))[1],
                (x,), (1.0,)
            )
            return u_t - eps * u_xx + 5.0 * u ** 3 - 5.0 * u
        return jnp.mean(vmap(pde_single)(t_res, x_res) ** 2)
    




# ============================================================
# Burgers: u_t + u * u_x = nu * u_xx
# ============================================================

class BurgersLoss:
    def __init__(self, model, nu=0.01):
        self.model = model
        self.nu = nu

    @staticmethod
    def ic_fn(x):
        return -jnp.sin(jnp.pi * x)

    def ic_loss(self, params, x_ic):
        B = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic], axis=-1)
        u_pred = self.model.apply(params, coords)[:, 0]
        return jnp.mean((u_pred - self.ic_fn(x_ic)) ** 2)

    def bc_loss(self, params, t_bc):
        def bc_single(t):
            u_l = self.model.apply(params, jnp.array([[t, -1.0]]))[0, 0]
            u_r = self.model.apply(params, jnp.array([[t, 1.0]]))[0, 0]
            return u_l ** 2 + u_r ** 2
        return jnp.mean(vmap(bc_single)(t_bc))

    def res_loss(self, params, t_res, x_res):
        nu = self.nu
        def u_net(t, x):
            return self.model.apply(params, jnp.array([[t, x]]))[0, 0]
        def pde_single(t, x):
            u, u_t = jax.jvp(lambda t_: u_net(t_, x), (t,), (1.0,))
            u_x = jax.jvp(lambda x_: u_net(t, x_), (x,), (1.0,))[1]
            _, u_xx = jax.jvp(
                lambda x_: jax.jvp(lambda x__: u_net(t, x__), (x_,), (1.0,))[1],
                (x,), (1.0,)
            )
            return u_t + u * u_x - nu * u_xx
        return jnp.mean(vmap(pde_single)(t_res, x_res) ** 2)


# ============================================================
# KdV: u_t + eta * u * u_x + mu^2 * u_xxx = 0
# ============================================================

class KdVLoss:
    def __init__(self, model, eta=1.0, mu=0.022):
        self.model = model
        self.eta = eta
        self.mu = mu

    @staticmethod
    def ic_fn(x):
        return jnp.cos(jnp.pi * x)

    def ic_loss(self, params, x_ic):
        B = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic], axis=-1)
        u_pred = self.model.apply(params, coords)[:, 0]
        return jnp.mean((u_pred - self.ic_fn(x_ic)) ** 2)

    def bc_loss(self, params, t_bc):
        def bc_single(t):
            def u_at(x_val):
                return self.model.apply(params, jnp.array([[t, x_val]]))[0, 0]
            u_l = u_at(-1.0)
            u_r = u_at(1.0)
            _, ux_l = jax.jvp(u_at, (-1.0,), (1.0,))
            _, ux_r = jax.jvp(u_at, (1.0,), (1.0,))
            return (u_l - u_r) ** 2 + (ux_l - ux_r) ** 2
        return jnp.mean(vmap(bc_single)(t_bc))

    def res_loss(self, params, t_res, x_res):
        eta, mu = self.eta, self.mu
        def u_net(t, x):
            return self.model.apply(params, jnp.array([[t, x]]))[0, 0]
        def pde_single(t, x):
            u, u_t = jax.jvp(lambda t_: u_net(t_, x), (t,), (1.0,))
            u_x = jax.jvp(lambda x_: u_net(t, x_), (x,), (1.0,))[1]
            _, u_xxx = jax.jvp(
                lambda x_: jax.jvp(
                    lambda x__: jax.jvp(
                        lambda x___: u_net(t, x___), (x__,), (1.0,)
                    )[1], (x_,), (1.0,)
                )[1], (x,), (1.0,)
            )
            return u_t + eta * u * u_x + mu ** 2 * u_xxx
        return jnp.mean(vmap(pde_single)(t_res, x_res) ** 2)


# ============================================================
# Channel Flow (NS): u_t + u*u_x + v*u_y = -p_x + (1/Re)*(u_xx + u_yy)
# ============================================================

class ChannelFlowLoss:
    def __init__(self, model, Re=100.0, L=5.0, H=1.0):
        self.model = model
        self.Re = Re
        self.L = L
        self.H = H

    @staticmethod
    def inlet_profile(y, H=1.0):
        return 1.5 * (1.0 - (2.0 * y / H) ** 2)

    def ic_loss(self, params, x_ic, y_ic):
        B = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic, y_ic], axis=-1)
        out = self.model.apply(params, coords)
        u_target = self.inlet_profile(y_ic, self.H)
        return jnp.mean((out[:, 0] - u_target) ** 2 + out[:, 1] ** 2)

    def bc_loss(self, params, t_bc, x_bc, y_bc):
        H, L = self.H, self.L

        coords_top = jnp.stack([t_bc, x_bc, jnp.full_like(x_bc, H/2)], axis=-1)
        out_top = self.model.apply(params, coords_top)
        loss_top = jnp.mean(out_top[:, 0] ** 2 + out_top[:, 1] ** 2)

        coords_bot = jnp.stack([t_bc, x_bc, jnp.full_like(x_bc, -H/2)], axis=-1)
        out_bot = self.model.apply(params, coords_bot)
        loss_bot = jnp.mean(out_bot[:, 0] ** 2 + out_bot[:, 1] ** 2)

        u_in = self.inlet_profile(y_bc, H)
        coords_in = jnp.stack([t_bc, jnp.zeros_like(y_bc), y_bc], axis=-1)
        out_in = self.model.apply(params, coords_in)
        loss_in = jnp.mean((out_in[:, 0] - u_in) ** 2 + out_in[:, 1] ** 2)

        coords_out = jnp.stack([t_bc, jnp.full_like(y_bc, L), y_bc], axis=-1)
        out_out = self.model.apply(params, coords_out)
        loss_out = jnp.mean(out_out[:, 2] ** 2)

        return loss_top + loss_bot + loss_in + loss_out

    def res_loss(self, params, t_res, x_res, y_res):
        Re = self.Re
        def uvp_net(t, x, y):
            return self.model.apply(params, jnp.array([[t, x, y]]))[0]
        def pde_single(t, x, y):
            uvp = uvp_net(t, x, y)
            u, v, p = uvp[0], uvp[1], uvp[2]
            u_t = grad(lambda t_: uvp_net(t_, x, y)[0])(t)
            v_t = grad(lambda t_: uvp_net(t_, x, y)[1])(t)
            u_x = grad(lambda x_: uvp_net(t, x_, y)[0])(x)
            u_y = grad(lambda y_: uvp_net(t, x, y_)[0])(y)
            v_x = grad(lambda x_: uvp_net(t, x_, y)[1])(x)
            v_y = grad(lambda y_: uvp_net(t, x, y_)[1])(y)
            p_x = grad(lambda x_: uvp_net(t, x_, y)[2])(x)
            p_y = grad(lambda y_: uvp_net(t, x, y_)[2])(y)
            u_xx = grad(grad(lambda x_: uvp_net(t, x_, y)[0]))(x)
            u_yy = grad(grad(lambda y_: uvp_net(t, x, y_)[0]))(y)
            v_xx = grad(grad(lambda x_: uvp_net(t, x_, y)[1]))(x)
            v_yy = grad(grad(lambda y_: uvp_net(t, x, y_)[1]))(y)
            r_x = u_t + u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy)
            r_y = v_t + u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy)
            r_c = u_x + v_y
            return r_x ** 2 + r_y ** 2 + r_c ** 2
        return jnp.mean(vmap(pde_single)(t_res, x_res, y_res))


# ============================================================
# Lid-Driven Cavity (NS)
# ============================================================

class LidDrivenCavityLoss:
    def __init__(self, model, Re=400.0):
        self.model = model
        self.Re = Re

    def ic_loss(self, params, x_ic, y_ic):
        B = x_ic.shape[0]
        coords = jnp.stack([jnp.zeros(B), x_ic, y_ic], axis=-1)
        out = self.model.apply(params, coords)
        return jnp.mean(out[:, 0] ** 2 + out[:, 1] ** 2)

    def bc_loss(self, params, t_bc, x_bc, y_bc):
        B = t_bc.shape[0]

        coords_top = jnp.stack([t_bc, x_bc, jnp.ones(B)], axis=-1)
        out_top = self.model.apply(params, coords_top)
        loss_top = jnp.mean((out_top[:, 0] - 1.0) ** 2 + out_top[:, 1] ** 2)

        coords_bot = jnp.stack([t_bc, x_bc, jnp.zeros(B)], axis=-1)
        out_bot = self.model.apply(params, coords_bot)
        loss_bot = jnp.mean(out_bot[:, 0] ** 2 + out_bot[:, 1] ** 2)

        coords_left = jnp.stack([t_bc, jnp.zeros(B), y_bc], axis=-1)
        out_left = self.model.apply(params, coords_left)
        loss_left = jnp.mean(out_left[:, 0] ** 2 + out_left[:, 1] ** 2)

        coords_right = jnp.stack([t_bc, jnp.ones(B), y_bc], axis=-1)
        out_right = self.model.apply(params, coords_right)
        loss_right = jnp.mean(out_right[:, 0] ** 2 + out_right[:, 1] ** 2)

        return loss_top + loss_bot + loss_left + loss_right

    def res_loss(self, params, t_res, x_res, y_res):
        Re = self.Re
        def uvp_net(t, x, y):
            return self.model.apply(params, jnp.array([[t, x, y]]))[0]
        def pde_single(t, x, y):
            uvp = uvp_net(t, x, y)
            u, v, p = uvp[0], uvp[1], uvp[2]
            u_t = grad(lambda t_: uvp_net(t_, x, y)[0])(t)
            v_t = grad(lambda t_: uvp_net(t_, x, y)[1])(t)
            u_x = grad(lambda x_: uvp_net(t, x_, y)[0])(x)
            u_y = grad(lambda y_: uvp_net(t, x, y_)[0])(y)
            v_x = grad(lambda x_: uvp_net(t, x_, y)[1])(x)
            v_y = grad(lambda y_: uvp_net(t, x, y_)[1])(y)
            p_x = grad(lambda x_: uvp_net(t, x_, y)[2])(x)
            p_y = grad(lambda y_: uvp_net(t, x, y_)[2])(y)
            u_xx = grad(grad(lambda x_: uvp_net(t, x_, y)[0]))(x)
            u_yy = grad(grad(lambda y_: uvp_net(t, x, y_)[0]))(y)
            v_xx = grad(grad(lambda x_: uvp_net(t, x_, y)[1]))(x)
            v_yy = grad(grad(lambda y_: uvp_net(t, x, y_)[1]))(y)
            r_x = u_t + u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy)
            r_y = v_t + u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy)
            r_c = u_x + v_y
            return r_x ** 2 + r_y ** 2 + r_c ** 2
        return jnp.mean(vmap(pde_single)(t_res, x_res, y_res))
