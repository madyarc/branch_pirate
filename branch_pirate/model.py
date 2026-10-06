import jax.numpy as jnp
import flax.linen as nn


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    activation: callable
    alpha_init: float = 0.0

    @nn.compact
    def __call__(self, x, u, v):
        identity = x
        f = self.activation(nn.Dense(self.hidden_dim)(x))
        z1 = f * u + (1 - f) * v
        g = self.activation(nn.Dense(self.hidden_dim)(z1))
        z2 = g * u + (1 - g) * v
        h = self.activation(nn.Dense(identity.shape[-1])(z2))
        alpha = self.param('alpha', nn.initializers.constant(self.alpha_init), (1,))
        return alpha * h + (1 - alpha) * identity


class GatedFourierEmbedding(nn.Module):
    n_bands: int = 4
    n_per_band: int = 16
    freq_min: float = 0.5
    freq_max: float = 20.0
    gate_init: float = 1.0

    @nn.compact
    def __call__(self, x):
        h = x[:, None]  # (B, 1)

        band_centers = jnp.logspace(
            jnp.log10(self.freq_min),
            jnp.log10(self.freq_max),
            self.n_bands
        )

        gate_logits = self.param(
            'gate_logits',
            nn.initializers.constant(self.gate_init),
            (self.n_bands,)
        )
        gates = nn.sigmoid(gate_logits)

        band_outputs = []
        for i in range(self.n_bands):
            k = self.param(
                f'band_{i}_freqs',
                lambda key, shape, center=band_centers[i]: center * jnp.linspace(0.5, 1.5, shape[0]),
                (self.n_per_band,)
            )
            proj = h * k[None, :]  # (B, n_per_band)
            band_feat = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
            band_outputs.append(gates[i] * band_feat)

        return jnp.concatenate(band_outputs, axis=-1)


class SpatialBranch(nn.Module):
    hidden_dim: int
    num_layers: int = 3
    activation: callable = nn.tanh
    n_bands: int = 4
    n_per_band: int = 16
    freq_min: float = 0.5
    freq_max: float = 20.0

    @nn.compact
    def __call__(self, x):
        h = GatedFourierEmbedding(
            n_bands=self.n_bands,
            n_per_band=self.n_per_band,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            name="gated_fourier"
        )(x)
        h = self.activation(nn.Dense(self.hidden_dim, name="fourier_proj")(h))
        for i in range(self.num_layers):
            h = self.activation(nn.Dense(self.hidden_dim, name=f"spatial_{i}")(h))
        return h


class TemporalBranch(nn.Module):
    hidden_dim: int
    num_layers: int = 3
    activation: callable = nn.tanh
    n_bands: int = 3
    n_per_band: int = 16
    freq_min: float = 0.5
    freq_max: float = 5.0

    @nn.compact
    def __call__(self, t):
        h = GatedFourierEmbedding(
            n_bands=self.n_bands,
            n_per_band=self.n_per_band,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            name="gated_fourier"
        )(t)
        h = self.activation(nn.Dense(self.hidden_dim, name="fourier_proj")(h))
        for i in range(self.num_layers):
            h = self.activation(nn.Dense(self.hidden_dim, name=f"temporal_{i}")(h))
        return h


class BranchNetModel(nn.Module):
    input_dim: int
    branch_t_hidden: int;  branch_x_hidden: int;  branch_y_hidden: int
    branch_layers: int;  pirate_hidden: int;  pirate_layers: int
    out_dim: int = 1;  activation: callable = nn.tanh;  alpha_init: float = 0.0
    spatial_n_bands: int = 4;   spatial_n_per_band: int = 16
    spatial_freq_min: float = 0.5;  spatial_freq_max: float = 20.0
    temporal_n_bands: int = 3;  temporal_n_per_band: int = 16
    temporal_freq_min: float = 0.5; temporal_freq_max: float = 5.0
    @nn.compact
    def __call__(self, coords):
        t_prime = TemporalBranch(hidden_dim=self.branch_t_hidden, num_layers=self.branch_layers,
            activation=self.activation, n_bands=self.temporal_n_bands, n_per_band=self.temporal_n_per_band,
            freq_min=self.temporal_freq_min, freq_max=self.temporal_freq_max, name="branch_t")(coords[:, 0])
        x_prime = SpatialBranch(hidden_dim=self.branch_x_hidden, num_layers=self.branch_layers,
            activation=self.activation, n_bands=self.spatial_n_bands, n_per_band=self.spatial_n_per_band,
            freq_min=self.spatial_freq_min, freq_max=self.spatial_freq_max, name="branch_x")(coords[:, 1])
        branches = [t_prime, x_prime]
        if self.input_dim == 3:
            y_prime = SpatialBranch(hidden_dim=self.branch_y_hidden, num_layers=self.branch_layers,
                activation=self.activation, n_bands=self.spatial_n_bands, n_per_band=self.spatial_n_per_band,
                freq_min=self.spatial_freq_min, freq_max=self.spatial_freq_max, name="branch_y")(coords[:, 2])
            branches.append(y_prime)
        z = jnp.concatenate(branches, axis=-1)
        z = self.activation(nn.Dense(self.pirate_hidden, name="input_proj")(z))
        u = self.activation(nn.Dense(self.pirate_hidden, name="Dense_u")(z))
        v = self.activation(nn.Dense(self.pirate_hidden, name="Dense_v")(z))
        for i in range(self.pirate_layers):
            z = PIModifiedBottleneck(hidden_dim=self.pirate_hidden, activation=self.activation,
                alpha_init=self.alpha_init, name=f"bottleneck_{i}")(z, u, v)
        out = nn.Dense(self.out_dim, name="final_output")(z)
        return out, z