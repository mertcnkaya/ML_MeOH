import os
import math
import joblib
import pandas as pd
import numpy as np
from math import sqrt, ceil
import matplotlib.pyplot as plt
import sys
import tkinter as tk

# =============================================================================
#                            USER-DEFINED INPUTS
# =============================================================================

# 1) Catalyst volume to distribute across channels
volume_input_cm3    = 25.8       # [cm³]
volume_input        = volume_input_cm3 * 1e-6  # → [m³]

# 2) Mixture (H₂/CO₂) flow & composition at STP
Q_mix_norm_NmLmin   = 9837.84       # [NmL/min] at STP (273.15K,1.01325bar)
y_CO2               = 0.17            # mole fractions of CO₂ and H₂
y_H2                = 0.83
# 3) Heat generated from the reaction in W
heat_reaction = 8.299
# 4) Reactor operating conditions
T_mix               = 520.0           # [K]  (convert to °C if your models used °C)
P_mix               = 40e5            # [Pa]

# 5) Packed‐bed parameters
epsilon             = 0.4             # void fraction
d_p                 = 1.5e-4          # particle diameter [m]

# 6) Channel geometry envelope parameters (all in mm)
w                   = 0.8             # wall thickness for real application [mm]
dist                = 0.5             # distance between channels [mm]
dist_hel            = 0.5
e                   = 22            # envelope width [mm]

# 7) Square‐geometry fixed aspect ratio and round‐sampling step
square_ratio        = 1.5
step_mm             = 0.1             # for sampling round diameters

# 8) Channel material parameters
lambda_realApp      = 15.0            # W/m·K  (real application)
lambda_test         = 0.3             # W/m·K  (test specimen)
w_test              = 2.0             # wall thickness test specimen [mm]

# =============================================================================
#               SUTHERLAND-LAW AND IDEAL-GAS DENSITY ROUTINES
# =============================================================================

R_UNIVERSAL = 8.314  # J/(mol·K)

SUTHER_PARAMS = {
    'N2':  (1.663e-5, 300.55, 111.0,   28.0134e-3),
    'CO2': (1.370e-5, 300.0,  240.0,   44.01e-3),
    'H2':  (8.761e-6, 293.15,  72.0,    2.016e-3)
}

def viscosity_sutherland(gas: str, T: float) -> float:
    mu_ref, T_ref, S, _ = SUTHER_PARAMS[gas]
    return mu_ref * (T / T_ref)**1.5 * ((T_ref + S) / (T + S))

def density_ideal_gas(gas: str, T: float, P: float) -> float:
    _, _, _, M = SUTHER_PARAMS[gas]
    return (P * M) / (R_UNIVERSAL * T)

# =============================================================================
#         HELPER: COMPUTE N₂ SETPOINT FROM MIXTURE REYNOLDS NUMBER
# =============================================================================

def compute_N2_setpoint_from_Re(
    Q_mix_norm_NmLmin, y_CO2, y_H2, T_mix, P_mix,
    crossA, epsilon, d_p, pipe_count
) -> float:
    # Norm & STP
    T_norm, P_norm = 273.15, 1.01325e5
    T_stp,  P_stp  = 293.15, 1.01325e5

    # 1) Mixture actual flow [m³/s]
    Q_act_mLmin = Q_mix_norm_NmLmin * (P_stp / P_mix) * (T_mix / T_stp)
    Q_act_m3s   = Q_act_mLmin / 1e6 / 60.0

    # 2) Mixture ρ & μ at (T_mix,P_mix)
    rho_CO2 = density_ideal_gas('CO2', T_mix, P_mix)
    rho_H2  = density_ideal_gas('H2',  T_mix, P_mix)
    mu_CO2  = viscosity_sutherland('CO2', T_mix)
    mu_H2   = viscosity_sutherland('H2',  T_mix)
    rho_mix = y_CO2 * rho_CO2 + y_H2 * rho_H2
    mu_mix  = y_CO2 * mu_CO2   + y_H2  * mu_H2

    # 3) Per-pipe superficial velocity & Re_p_mix
    Q_pipe   = Q_act_m3s / pipe_count
    u0_mix   = Q_pipe / (crossA * (1 - epsilon))
    Re_p_mix = rho_mix * u0_mix * d_p / mu_mix

    # 4) Back-calculate N2 norm velocity
    rho_N2_norm = density_ideal_gas('N2', T_norm, P_norm)
    mu_N2_norm  = viscosity_sutherland('N2', T_norm)
    u0_N2_norm  = Re_p_mix * mu_N2_norm / (rho_N2_norm * d_p)

    # 5) N2 flow per pipe at norm [m³/s] → NmL/min
    Q_N2_pipe_m3s    = u0_N2_norm * crossA * (1 - epsilon)
    Q_N2_pipe_NmLmin = Q_N2_pipe_m3s * 1e6 * 60.0

    return Q_N2_pipe_NmLmin * pipe_count

# =============================================================================
#                     LOAD ML MODELS, SCALERS, ENCODERS
# =============================================================================

rf_heat        = joblib.load("rf_model_kappa.pkl")
scaler_X_heat  = joblib.load("rf_scaler_X_kappa.pkl")
scaler_y_heat  = joblib.load("rf_scaler_y_kappa.pkl")
enc_heat       = joblib.load("rf_encoder_kappa.pkl")

rf_radial      = joblib.load("rf_radial_diff_model.pkl")
scaler_X_radial= joblib.load("rf_radial_diff_scaler_X.pkl")
scaler_y_radial= joblib.load("rf_radial_diff_scaler_y.pkl")
enc_radial     = joblib.load("rf_radial_diff_encoder.pkl")

rf_pressure    = joblib.load("rf_pressure_drop_model.pkl")
scaler_X_pressure = joblib.load("rf_pressure_drop_scaler_X.pkl")
scaler_y_pressure = joblib.load("rf_pressure_drop_scaler_y.pkl")
enc_pressure   = joblib.load("rf_pressure_drop_encoder.pkl")

rf_flow        = joblib.load("rf_avg_diff_model.pkl")
scaler_X_flow  = joblib.load("rf_avg_diff_scaler_X.pkl")
scaler_y_flow  = joblib.load("rf_avg_diff_scaler_y.pkl")
enc_flow       = joblib.load("rf_avg_diff_encoder.pkl")

cross_A_ranges_by_geom = {
    'round':       (7e-6, 2e-4),
    'square':      (7e-6, 2e-4),
    'round helix': (7.06e-6, 2.8e-5)
}
length_ranges_by_geom = {
    'round':       (0.05, 0.10),
    'square':      (0.05, 0.10),
    'round helix': (0.06, 0.07)
}

# =============================================================================
#                            SAFE ONE-HOT (NO CRASH)
# =============================================================================

def safe_ohe_batch(enc, labels):
    """
    One-hot rows using enc.categories_[0]; unseen labels become all-zeros.
    Keeps the exact column order/width expected by the trained model.
    """
    cats = list(enc.categories_[0])
    idx  = {c:i for i,c in enumerate(cats)}
    M = np.zeros((len(labels), len(cats)), dtype=float)
    for r, g in enumerate(labels):
        j = idx.get(g, None)
        if j is not None:
            M[r, j] = 1.0
    return M

# =============================================================================
#                         DESIGN-SPACE GENERATOR
# =============================================================================

_channel_cache = {}
def get_channel_count_round(d_mm):
    if d_mm in _channel_cache:
        return _channel_cache[d_mm]
    n = int(e / ((w + dist + d_mm/2)*sqrt(3)) * 2 - 1)
    m = ceil(e / ((w + dist + d_mm/2))) * 2 - 1
    cc= (n//2)*(ceil(m/2)-2)+((n//2)+1)*(ceil(m/2)-1)
    _channel_cache[d_mm] = cc
    return cc

_channel_hel_cache = {}
def get_channel_count_helix(d_mm):
    if d_mm in _channel_hel_cache:
        return _channel_hel_cache[d_mm]
    n = int(e / ((w + dist + d_mm/2)*sqrt(3)) * 2 - 1)
    m = ceil(e / ((w + dist + d_mm/2))) * 2 - 1
    cc= (n//2)*(ceil(m/2)-2)+((n//2)+1)*(ceil(m/2)-1)
    _channel_hel_cache[d_mm] = cc
    return cc

_square_cache = {}
def get_channel_count_square(a_mm, b_mm):
    key=(a_mm,b_mm)
    if key in _square_cache:
        return _square_cache[key]
    n = int(e / ((w + dist + a_mm/2)*sqrt(3)) * 2 - 1)
    m = ceil(e / ((w + dist + b_mm/2))) * 2 - 1
    cc= (n//2)*(ceil(m/2)-2)+((n//2)+1)*(ceil(m/2)-1)
    _square_cache[key] = cc
    return cc

def generate_design_space(volume):
    rows = []
    A_min_r,A_max_r = cross_A_ranges_by_geom['round']
    A_min_s,A_max_s = cross_A_ranges_by_geom['square']
    A_min_h,A_max_h = cross_A_ranges_by_geom['round helix']
    Lmin_r,Lmax_r = length_ranges_by_geom['round']
    Lmin_s,Lmax_s = length_ranges_by_geom['square']
    Lmin_h,Lmax_h = length_ranges_by_geom['round helix']

    # ROUND
    d_min = 2*sqrt(A_min_r/math.pi)*1000
    d_max = 2*sqrt(A_max_r/math.pi)*1000
    for d_mm in np.arange(d_min, d_max+1e-9, step_mm):
        area = math.pi*(d_mm/1000)**2/4
        cc = get_channel_count_round(d_mm)
        if cc<=0: continue
        length = volume/(area*cc)
        if Lmin_r<=length<=Lmax_r:
            rows.append({
                'Geometry-Type':'round','length (m)':length,
                'cross_A (m3)':area,'side_a (mm)':np.nan,
                'side_b (mm)':np.nan,'dimens (mm)':d_mm,
                'channel_count':cc
            })

    # SQUARE
    for crossA in np.linspace(A_min_s,A_max_s,num=100):
        b    = math.sqrt(crossA/square_ratio)
        a_mm = square_ratio*b*1e3
        b_mm = b*1e3
        area = (a_mm*b_mm)/1e6
        cc   = get_channel_count_square(a_mm,b_mm)
        if cc<=0: continue
        length=volume/(area*cc)
        if Lmin_s<=length<=Lmax_s:
            rows.append({
                'Geometry-Type':'square','length (m)':length,
                'cross_A (m3)':area,'side_a (mm)':a_mm,
                'side_b (mm)':b_mm,'dimens (mm)':np.nan,
                'channel_count':cc
            })

    # ROUND HELIX
    helix_dim = 12.0  # spiral diameter (fixed for all variants)
    d_min = 2 * math.sqrt(A_min_h / math.pi) * 1000
    d_max = 2 * math.sqrt(A_max_h / math.pi) * 1000

    for pipe_dim in np.arange(d_min, d_max + 1e-9, step_mm):
        spacing = helix_dim + pipe_dim  # spacing between centers of adjacent helices
        cc_h = get_channel_count_helix(spacing)
        if cc_h <= 0: continue
        pitch = 18.0
        area = math.pi * (pipe_dim / 1000)**2 / 4
        helix_length = volume / (area * cc_h)  # m
        turn_length_mm = math.sqrt((math.pi * helix_dim)**2 + pitch**2)  # mm
        axial_length   = helix_length * (pitch / turn_length_mm)         # m
        print(axial_length)
        if Lmin_h <= axial_length <= Lmax_h:
            rows.append({
                'Geometry-Type': 'round helix',
                'length (m)': axial_length,
                'cross_A (m3)': area,
                'side_a (mm)': np.nan,
                'side_b (mm)': np.nan,
                'dimens (mm)': pipe_dim,           # only the pipe diameter
                'channel_count': cc_h
            })
    return pd.DataFrame(rows)

# =============================================================================
#                             TOPSIS FUNCTION
# =============================================================================

def topsis(df, max_COls, min_cols, weights=None):
    cols = max_COls + min_cols
    D = df[cols].values.astype(float)
    R = np.zeros_like(D)
    for j in range(D.shape[1]):
        norm = math.sqrt((D[:,j]**2).sum())
        R[:,j] = D[:,j]/norm if norm else 0
    w = (np.ones(len(cols))/len(cols)) if weights is None else np.array([weights.get(c,0) for c in cols])
    w = w/w.sum()
    V = R * w
    best  = [ V[:,j].max() if c in max_COls else V[:,j].min() for j,c in enumerate(cols) ]
    worst = [ V[:,j].min() if c in max_COls else V[:,j].max() for j,c in enumerate(cols) ]
    Dp = np.sqrt(((V-best)**2).sum(axis=1))
    Dm = np.sqrt(((V-worst)**2).sum(axis=1))
    return Dm/(Dp+Dm+1e-12)

# =============================================================================
#                               CANDIDATE SELECTION UI
# =============================================================================

def select_candidate(df_top5, timeout_ms=60000):
    """
    Pops up a window letting the user select one of the df_top5 rows.
    Returns the selected row index (0-4). Defaults to 0 if no choice in timeout_ms.
    """
    root = tk.Tk()
    root.title("Select your preferred candidate")
    var = tk.IntVar(value=0)  # default = highest efficiency

    for idx, row in df_top5.iterrows():
        geom = row['Geometry-Type']
        if geom in ('round', 'round helix'):
            dim_text = f"diameter={row['diameter (mm)']:.3f} mm"
        else:
            dim_text = (f"a={row['side_a (mm)']:.3f} mm, "
                        f"b={row['side_b (mm)']:.3f} mm")

        text = (
            f"{idx+1}. {geom} | {dim_text} | "
            f"chan={row['channel_count']} | "
            f"eff={row['Efficiency (%)']:.1f}%"
        )
        tk.Radiobutton(root,
                       text=text,
                       variable=var,
                       value=idx,
                       anchor='w',
                       justify='left').pack(fill='x', padx=10, pady=2)

    def on_ok():
        root.quit()
    tk.Button(root, text="OK", command=on_ok).pack(pady=10)

    root.after(timeout_ms, on_ok)
    root.geometry("+%d+%d" % (root.winfo_screenwidth()//2 - 300,
                              root.winfo_screenheight()//2 - 200))
    root.mainloop()
    choice = var.get()
    root.destroy()
    return choice

# =============================================================================
#                               MAIN EXECUTION
# =============================================================================

if __name__=="__main__":
    step_mm = 0.1

    df_design = generate_design_space(volume_input)
    if df_design.empty:
        print("No valid combinations for volume =", volume_input_cm3, "cm³")
        sys.exit(0)

    df_design['N2_setpoint_NmLmin'] = df_design.apply(
        lambda r: compute_N2_setpoint_from_Re(
            Q_mix_norm_NmLmin, y_CO2, y_H2, T_mix, P_mix,
            r['cross_A (m3)'], epsilon, d_p, r['channel_count']
        ),
        axis=1
    )
    df_design['N2_per_channel'] = df_design['N2_setpoint_NmLmin'] / df_design['channel_count']

    n = len(df_design)
    geoms  = df_design['Geometry-Type'].values
    areas  = df_design['cross_A (m3)'].values
    lens   = df_design['length (m)'].values
    N2pc   = df_design['N2_per_channel'].values

    Tn2 = np.full(n, T_mix - 273.15)  # change to T_mix if trained in K
    Toil = np.full(n, 60.0)           # example oil temp; adjust to match training

    ohe_h = safe_ohe_batch(enc_heat, geoms)
    Xh_raw = np.hstack((ohe_h, areas.reshape(n,1), lens.reshape(n,1), N2pc.reshape(n,1), Tn2.reshape(n,1), Toil.reshape(n,1)))
    Xh = scaler_X_heat.transform(Xh_raw)

    ohe_r = safe_ohe_batch(enc_radial, geoms)
    Xr_raw = np.hstack((ohe_r, areas.reshape(n,1), lens.reshape(n,1), N2pc.reshape(n,1), Tn2.reshape(n,1), Toil.reshape(n,1)))
    Xr = scaler_X_radial.transform(Xr_raw)

    df_design['kappa_sim (W/m·K)'] = scaler_y_heat.inverse_transform(
        rf_heat.predict(Xh).reshape(-1,1)
    )[:,0]
    df_design['radial_diff (°C)']  = scaler_y_radial.inverse_transform(
        rf_radial.predict(Xr).reshape(-1,1)
    )[:,0]

    ohe_p = safe_ohe_batch(enc_pressure, geoms)
    Xp_raw = np.hstack((ohe_p, areas.reshape(n,1), lens.reshape(n,1), N2pc.reshape(n,1)))
    Xp     = scaler_X_pressure.transform(Xp_raw)

    ohe_f = safe_ohe_batch(enc_flow, geoms)
    Xf_raw = np.hstack((ohe_f, areas.reshape(n,1), lens.reshape(n,1), N2pc.reshape(n,1)))
    Xf     = scaler_X_flow.transform(Xf_raw)

    df_design['pressure_drop (bar)'] = scaler_y_pressure.inverse_transform(
        rf_pressure.predict(Xp).reshape(-1,1)
    )[:,0]
    df_design['avg_diff (m/s)'] = scaler_y_flow.inverse_transform(
        rf_flow.predict(Xf).reshape(-1,1)
    )[:,0]

    def lateral_area(row):
        geom = row['Geometry-Type']
        L = row['length (m)']
        if geom in ('round', 'round helix'):
            r = row['dimens (mm)'] / 1000.0 / 2.0  # m
            return 2 * math.pi * r * L
        else:  # square
            a = row['side_a (mm)'] / 1000.0
            b = row['side_b (mm)'] / 1000.0
            return 2 * (a + b) * L

    df_design['lateral_area (m2)'] = df_design.apply(lateral_area, axis=1)

    df_design['q_K (W)'] = (
        1.0 / (1.0/df_design['kappa_sim (W/m·K)'] + w/(lambda_realApp*1000.0) - w_test/(lambda_test*1000.0))
        * df_design['lateral_area (m2)']
        * df_design['channel_count']
    )

    df_design['delta_Q'] = heat_reaction - df_design['q_K (W)']

    df_design.to_csv("design_space.csv", index=False, float_format="%.12g")

    df_design = df_design[df_design['radial_diff (°C)'] <= 0.5]
    df_design = df_design[df_design['pressure_drop (bar)'] <= 1.0]
    df_design = df_design[df_design['avg_diff (m/s)']      <= 0.1]
    # df_design = df_design[df_design['delta_Q'] <= 0.0].reset_index(drop=True)

    if df_design.empty:
        print("No candidates remain after filtering.")
        sys.exit(0)

    df_small = df_design[[
        'Geometry-Type','kappa_sim (W/m·K)',
        'pressure_drop (bar)','length (m)',
        'dimens (mm)','side_a (mm)','side_b (mm)',
        'channel_count'
    ]].copy()
    weights = {
        'kappa_sim (W/m·K)'   : 0.30,
        'pressure_drop (bar)' : 0.10,
        'length (m)'          : 0.60
    }
    df_small['topsis_score'] = topsis(
        df_small,
        max_COls=['kappa_sim (W/m·K)'],
        min_cols=['pressure_drop (bar)','length (m)'],
        weights=weights
    )
    df_design['volume_from_dims'] = df_design['cross_A (m3)']*df_design['length (m)']*df_design['channel_count']

    df_small.assign(**{'Efficiency (%)': (df_small['topsis_score']*100).round(1)}) \
           .to_csv("candidates_predictions.csv", index=False)
    print(f"Exported {len(df_small)} candidates to 'candidates_predictions.csv'.")
    
    top5 = df_small.nlargest(5, 'topsis_score').reset_index(drop=True)
    final = pd.DataFrame({
        'Geometry-Type': top5['Geometry-Type'],
        'diameter (mm)' : top5['dimens (mm)'].round(3),
        'side_a (mm)'   : top5['side_a (mm)'].round(3),
        'side_b (mm)'   : top5['side_b (mm)'].round(3),
        'channel_count': top5['channel_count'],
        'Efficiency (%)': (top5['topsis_score']*100).round(1)
    })

    try:
        sel_idx = select_candidate(final)
    except Exception:
        # headless fallback → always pick the best (first) entry
        sel_idx = 0

    chosen = final.iloc[sel_idx]

    # Extract the key dims:
    geom = chosen['Geometry-Type']
    txt_path = "selected_candidate.txt"
    with open(txt_path, "w") as f:
        if geom == 'round':
            f.write(f"crossSec;CrossSectionInner_1.gh\n")
            f.write(f"flowPath;FlowPath_1.gh\n")
            f.write(f"skin;Skin_1.gh\n")
            f.write(f"diameter;{chosen['diameter (mm)']:.3f}\n")
        elif geom == 'round helix':
            f.write(f"crossSec;CrossSectionInner_1.gh\n")
            f.write(f"flowPath;FlowPath_2.gh\n")
            f.write(f"skin;Skin_1.gh\n")
            f.write(f"diameter;{chosen['diameter (mm)']:.3f}\n")
        else:  # square
            f.write(f"crossSec;CrossSectionInner_2.gh\n")
            f.write(f"flowPath;FlowPath_1.gh\n")
            f.write(f"skin;Skin_1.gh\n")
            f.write(f"side_b;{chosen['side_b (mm)']:.3f}\n")

