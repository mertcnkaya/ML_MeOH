import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt
import math

# === USER-DEFINED INPUT PARAMETER RANGES AND CONSTANTS ===

# Numeric feature ranges (per‐geometry lengths are defined separately)
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

# Ratio constant for square cross-section: a = square_ratio * b
square_ratio = 1.5  

# Valid target range for radial temperature difference (°C)
target_range = (0.0, 30.0)

# Number of synthetic candidates and suggestions
n_candidates  = 2000
n_suggestions = 5

# RandomForest hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_depth': [None, 10, 20, 40],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
}
# === END OF USER-DEFINED PARAMETERS ===




# === Load dataset ===
file_path = 'C:\\Users\\gh4617\\Desktop\\Preselection_tool\\experimental_results_v4.xlsx'
df = pd.read_excel(file_path, sheet_name='Heat_Transfer')

# === Select features and target column for radial temperature difference ===
selected_columns = [
    'Geometry-Type',
    'cross_A (in m3)',
    'length (m)',
    'N2_setpoint (mLn/min)',
    'T_in_N2 (°C)',
    'T_in_Oil (°C)'
]
df = df[selected_columns + ['Max_radial_difference_Outlet_Simulation (°C)']].dropna()

X_raw = df[selected_columns]
y_raw = df['Max_radial_difference_Outlet_Simulation (°C)'].values.reshape(-1, 1)

# === One-hot encode categorical feature ===
encoder = OneHotEncoder(sparse_output=False)
geometry_encoded = encoder.fit_transform(X_raw[['Geometry-Type']])
X_numeric = X_raw.drop(columns=['Geometry-Type']).values  # [cross_A, length, N2_setpoint, T_in_N2, T_in_Oil]
X_combined = np.hstack((geometry_encoded, X_numeric))

# === Normalize inputs and output ===
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_combined)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# === Random Forest with hyperparameter tuning ===
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=10,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train.ravel())

best_rf = grid_search.best_estimator_
print("Best RandomForest parameters:", grid_search.best_params_)

# === Evaluate best model ===
y_pred_test     = best_rf.predict(X_test)
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_test_inv      = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, y_pred_test_inv)
r2  = r2_score(y_test_inv, y_pred_test_inv)
# normalized MSE: divide by variance of the true targets
var_y_test = np.var(y_test_inv)
nmse       = mse / var_y_test
print(f"RandomForest MSE            : {mse:.4f}")
print(f"RandomForest R²             : {r2:.4f}")
print(f"Normalized MSE  : {nmse:.4f}")

# optional: normalized RMSE by range
nrmse = math.sqrt(mse) / (y_test_inv.max() - y_test_inv.min())
print(f"Normalized RMSE           : {nrmse:.4f}\n")

# calculate RMSE
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error (RMSE)          : {rmse:.4f}\n")

# === Save artifacts ===
joblib.dump(best_rf,   "rf_radial_diff_model.pkl")
joblib.dump(scaler_X,  "rf_radial_diff_scaler_X.pkl")
joblib.dump(scaler_y,  "rf_radial_diff_scaler_y.pkl")
joblib.dump(encoder,   "rf_radial_diff_encoder.pkl")

# === Visualization: Predicted vs True ===
y_pred_all     = best_rf.predict(X_scaled)
y_pred_all_inv = scaler_y.inverse_transform(y_pred_all.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_raw,          label='True Max_radial_difference_Outlet_Simulation (°C)',    marker='o')
plt.plot(y_pred_all_inv, label='Predicted Max_radial_difference_Outlet_Simulation (°C)', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Max_radial_difference_Outlet_Simulation (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Suggest NEW DATA POINTS (within defined ranges, with derived dimensions) ===
geom_labels = list(cross_A_ranges_by_geom.keys())
num_geom    = len(geom_labels)

candidates_scaled   = []
candidates_original = []

for _ in range(n_candidates):
    # 1) Randomly choose a geometry type
    geom = np.random.choice(geom_labels)

    # 2) Sample cross_A within that geometry's range
    crossA_min, crossA_max = cross_A_ranges_by_geom[geom]
    crossA = np.random.uniform(crossA_min, crossA_max)

    # 3) Sample length within geometry-specific range
    length_min, length_max = length_ranges_by_geom[geom]
    length = np.random.uniform(length_min, length_max)

    # 4) Sample other numeric features within their global ranges
    n2_min, n2_max         = input_ranges['N2_setpoint (mLn/min)']
    tin_n2_min, tin_n2_max = input_ranges['T_in_N2 (°C)']
    tin_oil_min, tin_oil_max = input_ranges['T_in_Oil (°C)']

    N2_setpoint = np.random.uniform(n2_min, n2_max)
    T_in_N2     = np.random.uniform(tin_n2_min, tin_n2_max)
    T_in_Oil    = np.random.uniform(tin_oil_min, tin_oil_max)

    # 5) One-hot encode the chosen geometry
    one_hot     = np.zeros(num_geom)
    geom_index  = geom_labels.index(geom)
    one_hot[geom_index] = 1.0

    # 6) Assemble full original feature vector
    numeric_vec   = np.array([crossA, length, N2_setpoint, T_in_N2, T_in_Oil])
    full_original = np.hstack((one_hot, numeric_vec))

    # 7) Scale this candidate using the fitted scaler_X
    full_scaled = scaler_X.transform(full_original.reshape(1, -1))[0]

    candidates_scaled.append(full_scaled)
    candidates_original.append({
        'Geometry-Type':        geom,
        'cross_A (in m3)':       crossA,
        'length (m)':            length,
        'N2_setpoint (mLn/min)': N2_setpoint,
        'T_in_N2 (°C)':          T_in_N2,
        'T_in_Oil (°C)':         T_in_Oil
    })

candidates_scaled = np.vstack(candidates_scaled)

# 8) Use NearestNeighbors on the REAL training set to find distance for each candidate
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(candidates_scaled)

# 9) Pick the candidates farthest from any real sample
farthest_indices  = np.argsort(distances.flatten())[::-1][:n_suggestions]
suggested_points = [candidates_original[i] for i in farthest_indices]

# 10) For each suggested point, compute derived dimensions
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

# 11) Display suggested new points with derived dimensions
suggestions_df = pd.DataFrame(enhanced_suggestions)
print("\n🔎 Suggested New Data Points (with derived dimensions):")
print(suggestions_df.to_string(index=False))

# === Prediction Function (with validation and cap) ===
def predict_radial_diff_rf(
    geometry_type,
    cross_A,
    length,
    N2_setpoint,
    T_in_N2,
    T_in_Oil,
    T_oil_max=60.0
):
    import numpy as np
    import pandas as pd
    import joblib

    model     = joblib.load("rf_radial_diff_model.pkl")
    scaler_X  = joblib.load("rf_radial_diff_scaler_X.pkl")
    scaler_y  = joblib.load("rf_radial_diff_scaler_y.pkl")
    encoder   = joblib.load("rf_radial_diff_encoder.pkl")

    geometry_labels = encoder.categories_[0]
    if geometry_type not in geometry_labels:
        raise ValueError(f"Invalid Geometry-Type: {geometry_type}. Must be one of {list(geometry_labels)}")

    # Validate cross_A range for this geometry
    geom_ranges = cross_A_ranges_by_geom.get(geometry_type)
    if geom_ranges:
        min_crossA, max_crossA = geom_ranges
        if not (min_crossA <= cross_A <= max_crossA):
            raise ValueError(
                f"cross_A {cross_A:.2e} out of range for geometry {geometry_type}: "
                f"[{min_crossA:.2e}, {max_crossA:.2e}]"
            )

    # Validate length range for this geometry
    length_ranges = length_ranges_by_geom.get(geometry_type)
    if length_ranges:
        min_len, max_len = length_ranges
        if not (min_len <= length <= max_len):
            raise ValueError(
                f"length {length:.2f} m out of range for geometry {geometry_type}: "
                f"[{min_len:.2f}, {max_len:.2f}]"
            )

    # Validate other numeric inputs
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

    geom_df = pd.DataFrame([[geometry_type]], columns=['Geometry-Type'])
    encoded_geometry = encoder.transform(geom_df)

    X_input = np.hstack((encoded_geometry, [[cross_A, length, N2_setpoint, T_in_N2, T_in_Oil]]))
    X_scaled_pred = scaler_X.transform(X_input)
    y_pred_scaled = model.predict(X_scaled_pred)
    y_pred_inv    = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

    # Cap within valid radial diff range
    capped = min(y_pred_inv, target_range[1])
    if capped < target_range[0]:
        capped = target_range[0]
    return capped

# === Example prediction for radial difference ===
example_radial_diff = predict_radial_diff_rf(
    geometry_type="round",
    cross_A=7.06858e-6,
    length=0.08,
    N2_setpoint=500,
    T_in_N2=22.1,
    T_in_Oil=60.0
)
print(f"\nExample Predicted Max_radial_difference_Outlet_Simulation (°C): {example_radial_diff:.5f} °C")
