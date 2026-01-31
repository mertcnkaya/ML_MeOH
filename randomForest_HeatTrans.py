import pandas as pd
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

# === USER-DEFINED INPUT PARAMETER RANGES AND CONSTANTS ===

# Numeric feature ranges (per‐geometry lengths are still defined separately)
input_ranges = {
    'N2_setpoint (mLn/min)': (100, 1500),
    'T_in_N2 (°C)':          (21.0, 25.0),
    'T_in_Oil (°C)':         (58.0, 61.0)
}

# cross_A valid ranges per Geometry-Type
cross_A_ranges_by_geom = {
    'round':       (7e-6, 2e-4),
    'square':      (7e-6, 2e-4),
    'round helix': (7.06e-6, 2.8e-5)
}

# length valid ranges per Geometry-Type
length_ranges_by_geom = {
    'round':       (0.05, 0.10),
    'square':      (0.05, 0.10),
    'round helix': (0.06, 0.07)
}

# Ratio constant for square cross‐section: a = square_ratio * b
square_ratio = 1.5  # Adjust as needed

# Number of synthetic candidates and suggestions
n_candidates  = 2000
n_suggestions = 5

# RandomForest hyperparameter grid (you can expand further if you like)
param_grid = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_depth': [None, 10, 20, 40],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
}
# === END OF USER-DEFINED PARAMETERS ===


# === 1. Load dataset ===
file_path =  //GIVE PATH HERE
df = pd.read_excel(file_path, sheet_name='Heat_Transfer')

# === 2. Select features and target “kappa” ===
selected_columns = [
    'Geometry-Type',
    'cross_A (in m3)',
    'length (m)',
    'N2_setpoint (mLn/min)',
    'T_in_N2 (°C)',
    'T_in_Oil (°C)'
]
df = df[selected_columns + ['kappa']].dropna()

X_raw = df[selected_columns]
y_raw = df['kappa'].values.reshape(-1, 1)  # shape = (n_samples, 1)


# === 3. One‐hot encode “Geometry-Type” ===
encoder = OneHotEncoder(sparse_output=False)
geom_ohe = encoder.fit_transform(X_raw[['Geometry-Type']])  # shape (n_samples, n_geom_categories)

X_numeric = X_raw.drop(columns=['Geometry-Type']).values  # shape (n_samples, 5)
X_all     = np.hstack((geom_ohe, X_numeric))               # final feature matrix

# === 4. Scale inputs and target ===

# 4.1 MinMax‐scale features into [0,1]
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_all)

# 4.2 STANDARD‐scale y (“kappa”) to mean=0, std=1
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_raw)


# === 5. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.20, random_state=42
)


# === 6. Random Forest with hyperparameter tuning (target = kappa) ===
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator   = rf,
    param_grid  = param_grid,
    cv          = 5,
    scoring     = 'neg_mean_squared_error',
    n_jobs      = -1,
    verbose     = 1
)
grid_search.fit(X_train, y_train.ravel())

best_rf = grid_search.best_estimator_
print("Best RandomForest parameters:", grid_search.best_params_)

# === 7. Evaluate best model on the hold‐out test set ===
y_pred_test_scaled = best_rf.predict(X_test).reshape(-1, 1)
y_pred_test        = scaler_y.inverse_transform(y_pred_test_scaled)  # back to original units
y_test_true        = scaler_y.inverse_transform(y_test)

mse   = mean_squared_error(y_test_true, y_pred_test)
r2    = r2_score(y_test_true, y_pred_test)
# normalized MSE: divide by variance of the true targets
var_y_test = np.var(y_test_true)
nmse       = mse / var_y_test
print(f"RandomForest MSE (kappa)           : {mse:.4f}")
print(f"RandomForest R² (kappa)             : {r2:.4f}")
print(f"Normalized MSE (MSE / Var[y_test_true]) : {nmse:.4f}")

# calculate RMSE
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error (RMSE)          : {rmse:.4f}\n")


# === 8. Save artifacts ===
joblib.dump(best_rf,         "rf_model_kappa.pkl")
joblib.dump(scaler_X,        "rf_scaler_X_kappa.pkl")
joblib.dump(scaler_y,        "rf_scaler_y_kappa.pkl")
joblib.dump(encoder,         "rf_encoder_kappa.pkl")


# === 9. Visualize Predicted vs True (kappa) over all training samples ===
y_pred_all_scaled = best_rf.predict(X_scaled).reshape(-1, 1)
y_pred_all        = scaler_y.inverse_transform(y_pred_all_scaled)

plt.figure(figsize=(10, 6))
plt.plot(y_raw,         label='True kappa',     marker='o')
plt.plot(y_pred_all,    label='Predicted kappa', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('kappa')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === 10. Suggest NEW DATA POINTS (within defined ranges, with derived dimensions) ===
geom_labels = list(cross_A_ranges_by_geom.keys())
num_geom    = len(geom_labels)

candidates_scaled   = []
candidates_original = []

for _ in range(n_candidates):
    # --- 10.1 Randomly choose a geometry type
    geom = np.random.choice(geom_labels)

    # --- 10.2 Sample cross_A within that geometry's range
    crossA_min, crossA_max = cross_A_ranges_by_geom[geom]
    crossA = np.random.uniform(crossA_min, crossA_max)

    # --- 10.3 Sample length within geometry-specific range
    length_min, length_max = length_ranges_by_geom[geom]
    length = np.random.uniform(length_min, length_max)

    # --- 10.4 Sample other numeric features within their global ranges
    n2_min, n2_max         = input_ranges['N2_setpoint (mLn/min)']
    tin_n2_min, tin_n2_max = input_ranges['T_in_N2 (°C)']
    tin_oil_min, tin_oil_max = input_ranges['T_in_Oil (°C)']

    N2_setpoint = np.random.uniform(n2_min, n2_max)
    T_in_N2     = np.random.uniform(tin_n2_min, tin_n2_max)
    T_in_Oil    = np.random.uniform(tin_oil_min, tin_oil_max)

    # --- 10.5 One‐hot encode this geometry
    one_hot     = np.zeros(num_geom)
    geom_index  = geom_labels.index(geom)
    one_hot[geom_index] = 1.0

    # --- 10.6 Assemble raw feature vector
    numeric_vec   = np.array([crossA, length, N2_setpoint, T_in_N2, T_in_Oil])
    raw_all       = np.hstack((one_hot, numeric_vec))

    # --- 10.7 Scale it with the fitted scaler_X
    scaled_all    = scaler_X.transform(raw_all.reshape(1, -1))[0]

    candidates_scaled.append(scaled_all)
    candidates_original.append({
        'Geometry-Type':        geom,
        'cross_A (in m3)':       crossA,
        'length (m)':            length,
        'N2_setpoint (mLn/min)': N2_setpoint,
        'T_in_N2 (°C)':          T_in_N2,
        'T_in_Oil (°C)':         T_in_Oil
    })

candidates_scaled = np.vstack(candidates_scaled)

# --- 10.8 Use NearestNeighbors to find “farthest from any real sample” ---
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(candidates_scaled)

# --- 10.9 Pick the top handful farthest away ---
farthest_indices  = np.argsort(distances.flatten())[::-1][:n_suggestions]
suggested_points = [candidates_original[i] for i in farthest_indices]

# --- 10.10 For each suggested point, compute derived dimensions ---
enhanced_suggestions = []
for pt in suggested_points:
    geom   = pt['Geometry-Type']
    crossA = pt['cross_A (in m3)']

    if geom in ['round', 'round helix']:
        # diameter = 2 * sqrt(area / π)
        diameter_mm = (2 * np.sqrt(crossA / np.pi)) * 1000
        enhanced_suggestions.append({
            **pt,
            'derived_diameter (mm)': diameter_mm
        })
    elif geom == 'square':
        # crossA = a * b with a = square_ratio * b => crossA = square_ratio * b^2
        b = np.sqrt(crossA / square_ratio)
        a = square_ratio * b
        enhanced_suggestions.append({
            **pt,
            'derived_a (mm)': a * 1000,
            'derived_b (mm)': b * 1000
        })
    else:
        enhanced_suggestions.append(pt)

# --- 10.11 Present those suggested points ---
suggestions_df = pd.DataFrame(enhanced_suggestions)
print("\n🔎 Suggested New Data Points (with derived dimensions):")
print(suggestions_df.to_string(index=False))


# === 11. Prediction Function (no clamping) ===
def predict_kappa(
    geometry_type,
    cross_A,
    length,
    N2_setpoint,
    T_in_N2,
    T_in_Oil
):
    """
    Loads the trained RF model + scalers + encoder (for kappa). 
    Returns the raw predicted kappa (not clamped).
    """
    import numpy as np
    import pandas as pd
    import joblib

    model       = joblib.load("rf_model_kappa.pkl")
    scaler_X_k  = joblib.load("rf_scaler_X_kappa.pkl")
    scaler_y_k  = joblib.load("rf_scaler_y_kappa.pkl")
    encoder_k   = joblib.load("rf_encoder_kappa.pkl")

    # 1) Validate geometry_type
    geometry_labels = encoder_k.categories_[0]
    if geometry_type not in geometry_labels:
        raise ValueError(f"Invalid Geometry-Type: {geometry_type}. Must be one of {list(geometry_labels)}")

    # 2) Validate cross_A range for that geometry
    geom_ranges = cross_A_ranges_by_geom.get(geometry_type)
    if geom_ranges:
        min_crossA, max_crossA = geom_ranges
        if not (min_crossA <= cross_A <= max_crossA):
            raise ValueError(
                f"cross_A {cross_A:.2e} out of range for geometry {geometry_type}: "
                f"[{min_crossA:.2e}, {max_crossA:.2e}]"
            )

    # 3) Validate length for that geometry
    length_ranges = length_ranges_by_geom.get(geometry_type)
    if length_ranges:
        min_len, max_len = length_ranges
        if not (min_len <= length <= max_len):
            raise ValueError(
                f"length {length:.2f} m out of range for geometry {geometry_type}: "
                f"[{min_len:.2f}, {max_len:.2f}]"
            )

    # 4) Validate other numeric inputs
    all_inputs = {
        'N2_setpoint (mLn/min)': N2_setpoint,
        'T_in_N2 (°C)':          T_in_N2,
        'T_in_Oil (°C)':         T_in_Oil
    }
    for param, val in all_inputs.items():
        mn, mx = input_ranges.get(param, (None, None))
        if mn is not None and mx is not None:
            if not (mn <= val <= mx):
                raise ValueError(f"{param} {val:.2e} is outside valid range: [{mn:.2e}, {mx:.2e}]")

    # 5) One‐hot encode geometry_type
    geom_df = pd.DataFrame([[geometry_type]], columns=['Geometry-Type'])
    encoded_geometry = encoder_k.transform(geom_df)

    # 6) Build raw feature vector exactly as during training
    X_row = np.hstack((encoded_geometry, [[cross_A, length, N2_setpoint, T_in_N2, T_in_Oil]]))

    # 7) Scale with scaler_X_k
    X_scaled_pred = scaler_X_k.transform(X_row)

    # 8) Predict (in scaled‐y space) and invert scale
    y_pred_scaled = model.predict(X_scaled_pred)
    y_pred_inv    = scaler_y_k.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

    return y_pred_inv


# === 12. Example prediction (should be near “48” if your Excel says so) ===
example_result = predict_kappa(
    geometry_type="round",
    cross_A=7.06858e-6,
    length=0.05,
    N2_setpoint=500,
    T_in_N2=22.98,
    T_in_Oil=60.81
)
print(f"\nExample Predicted kappa (with validation): {example_result:.4f}")

