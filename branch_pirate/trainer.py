import jax
import jax.numpy as jnp
from jax import jit, vmap, tree_util
from jax.flatten_util import ravel_pytree
import optax
import time


@jit
def tree_dot(a, b):
    return jnp.dot(ravel_pytree(a)[0], ravel_pytree(b)[0])

@jit
def tree_norm(a):
    flat = ravel_pytree(a)[0]
    return jnp.sqrt(jnp.dot(flat, flat))

@jit
def flatten_grads(grads_list):
    stacked = tree_util.tree_map(lambda *g: jnp.stack(g), *grads_list)
    return vmap(lambda g: ravel_pytree(g)[0])(stacked)

@jit
def solve_cagrad_weighted(grads_flat, loss_weights, c=0.9, num_steps=20, step_size=25.0):
    n_grads = grads_flat.shape[0]
    weights_norm = loss_weights / jnp.sum(loss_weights)

    g0 = weights_norm @ grads_flat
    g0_norm = jnp.linalg.norm(g0) + 1e-10
    phi = (c * g0_norm) ** 2
    GG = grads_flat @ grads_flat.T
    Gg0 = grads_flat @ g0

    w = jnp.ones(n_grads) / n_grads

    def cagrad_step(w, _):
        g_w_norm_sq = jnp.dot(w, jnp.dot(GG, w))
        g_w_norm = jnp.sqrt(g_w_norm_sq + 1e-10)
        term_2 = (jnp.sqrt(phi) / (g_w_norm + 1e-10)) * jnp.dot(GG, w)
        grad_F = Gg0 + term_2
        w_unconstrained = w - step_size * grad_F
        w_proj = jnp.maximum(w_unconstrained, 0)
        w_new = w_proj / (jnp.sum(w_proj) + 1e-10)
        return w_new, None

    w_opt, _ = jax.lax.scan(cagrad_step, w, None, length=num_steps)

    g_w_opt = w_opt @ grads_flat
    g_w_opt_norm = jnp.linalg.norm(g_w_opt) + 1e-10
    d_cagrad = g0 + (jnp.sqrt(phi) / g_w_opt_norm) * g_w_opt
    return w_opt, d_cagrad


class Trainer1D:
    """For 1D PDEs: Allen-Cahn, Burgers, KdV — inputs (t, x)"""
    def __init__(self, model, loss_fn, tx):
        self.model = model
        self.loss_fn = loss_fn
        self.tx = tx
        self._build_step()

    def _build_step(self):
        loss_fn = self.loss_fn
        tx = self.tx

        @jit
        def step(params, opt_state, x_ic, t_bc, t_res, x_res, w_ic, w_bc, cagrad_c):
            l_ic, g_ic = jax.value_and_grad(lambda p: loss_fn.ic_loss(p, x_ic))(params)
            l_bc, g_bc = jax.value_and_grad(lambda p: loss_fn.bc_loss(p, t_bc))(params)
            l_res, g_res = jax.value_and_grad(lambda p: loss_fn.res_loss(p, t_res, x_res))(params)

            grads_flat = flatten_grads([g_ic, g_bc, g_res])
            loss_weights = jnp.array([w_ic, w_bc, 1.0])
            w_opt, d_cagrad = solve_cagrad_weighted(grads_flat, loss_weights, c=cagrad_c)

            _, unflatten_fn = ravel_pytree(params)


            updates, new_opt = tx.update(unflatten_fn(d_cagrad), opt_state, params)
            new_params = optax.apply_updates(params, updates)
            grad_norm = jnp.sqrt(jnp.dot(d_cagrad, d_cagrad))

            return new_params, new_opt, l_ic, l_bc, l_res, grad_norm

        self.step = step

    def warmup(self, params, opt_state, sampler, cfg):
        """Trigger JIT compilation before training loop."""
        print("JIT compiling training step...", flush=True)
        t0 = time.time()
        x_ic, t_bc, t_res, x_res = sampler.sample(
            cfg['batch_ic'], cfg['batch_bc'], cfg['batch_res'],
            beta_b=cfg['beta_b_start'])
        w = jnp.array(0.0)
        c = jnp.array(cfg['cagrad_c_start'])
        # Run one step to trigger compilation
        result = self.step(params, opt_state, x_ic, t_bc, t_res, x_res, w, w, c)
        # Block until compilation + execution finishes
        jax.block_until_ready(result)
        t1 = time.time()
        print(f"JIT compilation done in {t1-t0:.1f}s", flush=True)
        # Return original params/opt_state — discard the warm-up step
        return params, opt_state


class TrainerW1D:
    """For 1D wave equation: u_tt = c^2 u_xx, inputs (t, x)"""
    def __init__(self, model, loss_fn, tx):
        self.model   = model
        self.loss_fn = loss_fn
        self.tx      = tx
        self._build_step()

    def _build_step(self):
        loss_fn = self.loss_fn
        tx      = self.tx

        @jit
        def step(params, opt_state, x_ic, t_bc, t_res, x_res,
                 w_ic, w_icv, w_bc, cagrad_c):
            l_ic,  g_ic  = jax.value_and_grad(lambda p: loss_fn.ic_loss(p, x_ic))(params)
            l_icv, g_icv = jax.value_and_grad(lambda p: loss_fn.ic_vel_loss(p, x_ic))(params)
            l_bc,  g_bc  = jax.value_and_grad(lambda p: loss_fn.bc_loss(p, t_bc))(params)
            l_res, g_res = jax.value_and_grad(lambda p: loss_fn.res_loss(p, t_res, x_res))(params)

            grads_flat   = flatten_grads([g_ic, g_icv, g_bc, g_res])
            loss_weights = jnp.array([w_ic, w_icv, w_bc, 1.0])

            w_opt, d_cagrad = solve_cagrad_weighted(grads_flat, loss_weights, c=cagrad_c)

            _, unflatten_fn = ravel_pytree(params)
            updates, new_opt = tx.update(unflatten_fn(d_cagrad), opt_state, params)
            new_params = optax.apply_updates(params, updates)
            grad_norm  = jnp.sqrt(jnp.dot(d_cagrad, d_cagrad))

            return new_params, new_opt, l_ic, l_icv, l_bc, l_res, grad_norm

        self.step = step

    def warmup(self, params, opt_state, sampler, cfg):
        """Trigger JIT compilation before training loop."""
        print("JIT compiling training step...", flush=True)
        t0 = time.time()
        x_ic, t_bc, t_res, x_res = sampler.sample(
            cfg['batch_ic'], cfg['batch_bc'], cfg['batch_res'],
            beta_b=cfg['beta_b_start'])
        w = jnp.array(0.0)
        c = jnp.array(cfg['cagrad_c_start'])
        result = self.step(params, opt_state, x_ic, t_bc, t_res, x_res, w, w, w, c)
        jax.block_until_ready(result)
        t1 = time.time()
        print(f"JIT compilation done in {t1-t0:.1f}s", flush=True)
        return params, opt_state
