from bioelec_pinn import Parameters

CYCLE_S_IN = {1: 1.000, 2: 1.651, 3: 2.436, 4: 3.619}  # [g/L]
CYCLE_S_FINAL = {1: 0.223, 2: 0.266, 3: 0.338, 4: 0.414}  # [g/L], for reference
CYCLE_ALPHA_0 = {1: 0.3, 2: 0.70, 3: 0.99, 4: 0.995}  # Table 4
CYCLE_DURATION = {  # Table 3: 96, 131, 240, 383 h
    1: 96.0 / 24.0,
    2: 131.0 / 24.0,
    3: 240.0 / 24.0,
    4: 383.0 / 24.0,
}

# Table 5 (Cycle 1) + Table 4 initial conditions.
CYCLE_PARAMETERS = {
    1: Parameters(
        q_max_p=0.292,
        mu_max_p=0.12,
        q_max_s=0.318,
        mu_max_s=1.29,
        K_Ss=0.255,
        K_M=0.0102,
        Y_med=49.2,
        k_mt=1.01,
        K_r1=600.4,
        K_r2=12.21,
    ),
    2: Parameters(
        q_max_p=0.618,
        mu_max_p=0.198,
        q_max_s=0.321,
        mu_max_s=1.006,
        K_Ss=0.945,
        K_M=0.82,
        Y_med=41.86,
        k_mt=3.147,
        K_r1=314.5,
        K_r2=38.85,
    ),
    3: Parameters(
        q_max_p=4.341,
        mu_max_p=2.163,
        q_max_s=4.125,
        mu_max_s=0.865,
        K_Ss=5.69,
        K_M=0.395,
        Y_med=18.56,
        k_mt=4.263,
        K_r1=1714.3,
        K_r2=7.459,
    ),
    4: Parameters(
        q_max_p=2.856,
        mu_max_p=3.366,
        q_max_s=1.861,
        mu_max_s=0.313,
        K_Ss=5.112,
        K_M=0.414,
        Y_med=16.07,
        k_mt=5.456,
        K_r1=405.4,
        K_r2=15.0,
    ),
}
