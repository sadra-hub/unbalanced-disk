import GPy
import numpy as np

def simulate_narx_model(u_sequence, y_initial, gp_model, na, nb):
    y_simulated = y_initial[:50].copy()
    prediction_variances = [0] * 50

    past_u = u_sequence[50 - nb:50].copy()
    past_y = y_simulated[-na:].copy()

    for u_current in u_sequence[50:]:
        input_vector = np.concatenate([past_u, past_y]).reshape(1, -1)
        mean, variance = gp_model.predict(input_vector)
        y_pred = mean.item()

        past_u.append(u_current)
        past_u.pop(0)

        past_y.append(y_pred)
        past_y.pop(0)

        y_simulated.append(y_pred)
        prediction_variances.append(variance.item())

    return np.array(y_simulated)

def construct_io_dataset(u_data, y_data, na, nb):
    inputs, targets = [], []
    for k in range(max(na, nb), len(y_data)):
        io_vector = np.concatenate([u_data[k - nb:k], y_data[k - na:k]])
        inputs.append(io_vector)
        targets.append(y_data[k])
    return np.array(inputs), np.array(targets)

# Load data
npz_data = np.load('disc-benchmark-files/training-val-test-data.npz')
th_all = npz_data['th']
u_all = npz_data['u']

train_split = int(0.8 * len(th_all))
valid_split = int(0.9 * len(th_all))

th_train, th_valid, th_test = th_all[:train_split], th_all[train_split:valid_split], th_all[valid_split:]
u_train, u_valid, u_test = u_all[:train_split], u_all[train_split:valid_split], u_all[valid_split:]

# Hyperparameter grid search
na_nb_candidates = [2, 5, 8, 11]
inducing_point_counts = [300, 100, 75, 50]

for num_inducing in inducing_point_counts:
    best_rmse_pred, best_rmse_sim = float('inf'), float('inf')
    best_pred_config, best_sim_config = (0, 0), (0, 0)

    for na in na_nb_candidates:
        for nb in na_nb_candidates:
            X_train, y_train = construct_io_dataset(u_train, th_train, na, nb)
            X_valid, y_valid = construct_io_dataset(u_valid, th_valid, na, nb)

            y_train = y_train.reshape(-1, 1)
            y_valid = y_valid.reshape(-1, 1)

            Z = np.random.uniform(-3, 3, size=(num_inducing, X_train.shape[1]))
            kernel = GPy.kern.RBF(input_dim=X_train.shape[1]) + GPy.kern.White(input_dim=X_train.shape[1])

            gp_model = GPy.models.SparseGPRegression(X_train, y_train, kernel=kernel, Z=Z)
            gp_model.optimize('bfgs')

            y_pred, _ = gp_model.predict(X_valid)
            rmse_pred = np.mean(np.sqrt((y_valid - y_pred) ** 2))

            y_sim = simulate_narx_model(list(u_valid), list(th_valid), gp_model, na, nb)
            rmse_sim = np.sqrt(np.mean((y_sim - th_valid) ** 2))

            if rmse_pred < best_rmse_pred:
                best_rmse_pred = rmse_pred
                best_pred_config = (na, nb)

            if rmse_sim < best_rmse_sim:
                best_rmse_sim = rmse_sim
                best_sim_config = (na, nb)

    print(f"Inducing: {num_inducing}, Best Prediction na:{best_pred_config[0]}, nb:{best_pred_config[1]}, RMSE: {best_rmse_pred:.4f}")
    print(f"Inducing: {num_inducing}, Best Simulation na:{best_sim_config[0]}, nb:{best_sim_config[1]}, RMSE: {best_rmse_sim:.4f}")

# Final model training and testing (based on best parameters found)
na, nb, inducing = 8, 2, 1000
X_train, y_train = construct_io_dataset(u_train, th_train, na, nb)
X_valid, y_valid = construct_io_dataset(u_valid, th_valid, na, nb)
X_test, y_test = construct_io_dataset(u_test, th_test, na, nb)

input_dim = X_train.shape[1]
Z = np.random.uniform(-3, 3, size=(inducing, input_dim))
kernel = GPy.kern.RBF(input_dim=input_dim) + GPy.kern.White(input_dim=input_dim, variance=1.)

model = GPy.models.SparseGPRegression(X_train, y_train.reshape(-1, 1), kernel=kernel, Z=Z)
model.optimize('bfgs')

# Validation
y_valid_pred, _ = model.predict(X_valid)
rmse_valid_pred = np.mean(np.sqrt((y_valid.reshape(-1, 1) - y_valid_pred) ** 2))
print(f"Validation Prediction RMSE: {rmse_valid_pred:.4f}")

sim_valid = simulate_narx_model(list(u_valid), list(th_valid), model, na, nb)
rmse_valid_sim = np.sqrt(np.mean((sim_valid - th_valid) ** 2))
print(f"Validation Simulation RMSE: {rmse_valid_sim:.4f}")

# Test
sim_test = simulate_narx_model(list(u_test), list(th_test), model, na, nb)
rmse_test_sim = np.sqrt(np.mean((sim_test - th_test) ** 2))
print(f"Test Simulation RMSE: {rmse_test_sim:.4f}")

y_test_pred, _ = model.predict(X_test)
rmse_test_pred = np.mean(np.sqrt((y_test.reshape(-1, 1) - y_test_pred) ** 2))
print(f"Test Prediction RMSE: {rmse_test_pred:.4f}")

# Hidden test prediction
hidden_data = np.load('disc-benchmark-files/hidden-test-prediction-submission-file.npz')
upast_hidden = hidden_data['upast']
thpast_hidden = hidden_data['thpast']

X_hidden = np.concatenate([upast_hidden[:, 15 - nb:], thpast_hidden[:, 15 - na:]], axis=1)

y_hidden_pred, _ = model.predict(X_hidden)
assert y_hidden_pred.shape[0] == upast_hidden.shape[0], "Sample size mismatch in hidden test!"

np.savez('disc-submission-files/sparse-gp-hidden-test-prediction-submission-file.npz', upast=upast_hidden, thpast=thpast_hidden, thnow=y_hidden_pred)