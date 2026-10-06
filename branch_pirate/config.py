DEFAULT_CONFIG = {
    'n_steps': 150000,
    'batch_ic': 256,
    'batch_bc': 256,
    'batch_res': 1024,
    'warmup_icbc': 5000,
    'learning_rate': 1e-3,
    'warmup_steps': 2000,
    'transition_steps': 5000,
    'decay_rate': 0.9,
    'lr_end': 1e-7,
    'log_every': 1000,
    'print_every': 10000,
    'seed_sampler': 1,
    'beta_b_start': 10.0,
    'beta_b_end': 1.0,
    'beta_anneal_steps': 10000,
    'cagrad_c_start': 0.80,
    'cagrad_c_end': 0.0,
    'cagrad_c_decay_frac': 0.8,
    # Architecture
    'input_dim': 2,
    'out_dim': 1,
    'branch_t_hidden': 64,
    'branch_x_hidden': 64,
    'branch_y_hidden': 64,
    'branch_layers': 2,
    'pirate_hidden': 32,
    'pirate_layers': 5,
    'alpha_init': 0.0,
    # Gated Fourier — spatial
    'spatial_n_bands': 4,
    'spatial_n_per_band': 8,
    'spatial_freq_min': 0.5,
    'spatial_freq_max': 20.0,
    # Gated Fourier — temporal
    'temporal_n_bands': 3,
    'temporal_n_per_band': 8,
    'temporal_freq_min': 0.5,
    'temporal_freq_max': 5.0,
}



DEFAULT_CONFIG_V1 = {
    'n_steps': 150000,
    'batch_ic': 256,
    'batch_bc': 256,
    'batch_res': 1024,
    'warmup_icbc': 5000,
    'learning_rate': 1e-3,
    'warmup_steps': 2000,
    'transition_steps': 5000,
    'decay_rate': 0.9,
    'lr_end': 1e-7,
    'log_every': 1000,
    'print_every': 10000,
    'seed_sampler': 1,
    'beta_b_start': 10.0,
    'beta_b_end': 1.0,
    'beta_anneal_steps': 80000,
    'cagrad_c_start': 0.80,
    'cagrad_c_end': 0.0,
    'cagrad_c_decay_frac': 0.8,
    # Architecture — branches
    'input_dim': 2,
    'out_dim': 1,
    'branch_t_hidden': 16,
    'branch_x_hidden': 32,
    'branch_y_hidden': 64,
    'branch_layers': 2,
    # IC block
    'ic_hidden': 32,
    'ic_layers': 2,
    # Physics block
    'phy_hidden': 32,
    'phy_layers': 3,
    # Fusion (PirateNet)
    'pirate_hidden': 32,
    'pirate_layers': 5,
    'alpha_init': 0.0,
    # Gated Fourier — spatial
    'spatial_n_bands': 4,
    'spatial_n_per_band': 8,
    'spatial_freq_min': 0.5,
    'spatial_freq_max': 20.0,
    # Gated Fourier — temporal
    'temporal_n_bands': 3,
    'temporal_n_per_band': 8,
    'temporal_freq_min': 0.5,
    'temporal_freq_max': 5.0,
}



DEFAULT_CONFIG_V2 = {
    'n_steps': 200000,
    'batch_ic': 256,
    'batch_bc': 256,
    'batch_res': 2048,
    'warmup_icbc': 5000,
    'learning_rate': 1e-3,
    'warmup_steps': 2000,
    'transition_steps': 5000,
    'decay_rate': 0.9,
    'lr_end': 1e-7,
    'log_every': 1000,
    'print_every': 10000,
    'seed_sampler': 1,
    'beta_b_start': 10.0,
    'beta_b_end': 1.0,
    'beta_anneal_steps': 80000,
    # Architecture — branches
    'input_dim': 2,
    'out_dim': 1,
    'branch_t_hidden': 16,
    'branch_x_hidden': 32,
    'branch_y_hidden': 64,
    'branch_layers': 2,
    # IC block
    'ic_hidden': 32,
    'ic_layers': 2,
    # Physics block (with PirateNet bottleneck)
    'phy_hidden': 32,
    'phy_layers': 3,
    'alpha_init': 0.0,

    # Gated Fourier — spatial
    'spatial_n_bands': 4,
    'spatial_n_per_band': 8,
    'spatial_freq_min': 0.5,
    'spatial_freq_max': 20.0,
    # Gated Fourier — temporal
    'temporal_n_bands': 3,
    'temporal_n_per_band': 8,
    'temporal_freq_min': 0.5,
    'temporal_freq_max': 5.0,
}

# ============================================================
# Add to config.py
# ============================================================

DEFAULT_CONFIG_V3 = {
    'n_steps': 200000,
    'batch_ic': 256,
    'batch_bc': 256,
    'batch_res': 2048,
    'warmup_icbc': 5000,
    'learning_rate': 1e-3,
    'warmup_steps': 2000,
    'transition_steps': 5000,
    'decay_rate': 0.9,
    'lr_end': 1e-7,
    'log_every': 1000,
    'print_every': 10000,
    'seed_sampler': 1,
    'beta_b_start': 10.0,
    'beta_b_end': 1.0,
    'beta_anneal_steps': 80000,
    'cagrad_c_start': 0.80,
    'cagrad_c_decay_frac': 0.8,
    # Architecture
    'input_dim': 2,
    'out_dim': 1,
    'branch_t_hidden': 16,
    'branch_x_hidden': 32,
    'branch_y_hidden': 64,
    'branch_layers': 2,
    'pirate_hidden': 32,
    'pirate_layers': 5,
    'alpha_init': 0.0,
    'eps': 0.0001,
    # Gated Fourier — spatial
    'spatial_n_bands': 4,
    'spatial_n_per_band': 8,
    'spatial_freq_min': 0.5,
    'spatial_freq_max': 20.0,
    # Gated Fourier — temporal
    'temporal_n_bands': 3,
    'temporal_n_per_band': 8,
    'temporal_freq_min': 0.5,
    'temporal_freq_max': 5.0,
}

# ============================================================
# Add to config.py
# ============================================================

DEFAULT_CONFIG_V4 = {
    'n_steps': 200000,
    'batch_ic': 256,
    'batch_bc': 256,
    'batch_res': 2048,
    'warmup_icbc': 5000,
    'learning_rate': 1e-3,
    'warmup_steps': 2000,
    'transition_steps': 5000,
    'decay_rate': 0.9,
    'lr_end': 1e-7,
    'log_every': 1000,
    'print_every': 10000,
    'seed_sampler': 1,
    'beta_b_start': 10.0,
    'beta_b_end': 1.0,
    'beta_anneal_steps': 80000,
    'cagrad_c_start': 0.80,
    'cagrad_c_decay_frac': 0.8,
    # Architecture
    'out_dim': 1,
    'branch_t_hidden': 16,
    'branch_x_hidden': 16,
    'branch_layers': 2,
    'pirate_hidden': 32,
    'pirate_layers': 5,
    'alpha_init': 0.0,
    'eps': 0.0001,
    # Gated Fourier — temporal
    'temporal_n_bands': 3,
    'temporal_n_per_band': 8,
    'temporal_freq_min': 0.5,
    'temporal_freq_max': 5.0,
}