from parc import model
from parc import IO
from parc import losses
from parc import graphs
import keras
import numpy as np

# initialize input data characteristics
data_dir = (
    "C:\\Users\\Austin Leonard\\parc_data\\data\\raw"  # input data file directory
)
case_numbers = 42  # number of cases
del_t = 0.79  # time step size (ns)
time_steps = 19  # desired time steps for data processing
t_idx = del_t * np.arange(1, time_steps + 1)  # time index in nanoseconds
for i in range(len(t_idx)):
    t_idx[i] = round(t_idx[i], 2)

# initialize adaptable model parameters
input_size = 240
n_fields = 1
n_timesteps = 19
numFeatureMaps = 258

# initialize loss parameters
weight_loss = [4, 1, 1]  # initial timestep weights, middle weights, late weights

# Parse the raw data and return microstructure data and temperature/pressure outputs
data_in, total_output, initial_vals, normalizing_constants = IO.parse_data(
    data_dir, time_steps, (del_t * (10**-9))
)

np.save("output_data", total_output)
np.save("input_data", data_in)
np.save("initial_vals", initial_vals)
np.save("normalizing_constants.npy", normalizing_constants)

total_output = np.load("output_data.npy")
data_in = np.load("input_data.npy")
initial_vals = np.load("initial_vals.npy")
normalizing_constants = np.load("normalizing_constants.npy", allow_pickle=True)

# split data set
(
    X_train,
    y_train,
    X_train_init,
    X_val,
    y_val,
    X_val_init,
    test_X,
    test_Y,
    test_X_init,
) = IO.split_data(data_in, total_output, initial_vals, splits=[0.6, 0.2, 0.2])

# initilize model parameters
parc_model = model.PARC(
    input_size, n_fields, n_timesteps=time_steps, n_featuremaps=numFeatureMaps
)
parc_model.build()
loss = ["mse", "mse"]
loss_weights = [1, 1]
optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
metrics = ["acc"]
parc_model.compile(
    loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics
)

# reshape to match old data format
y_train = IO.reshape_old(y_train)
y_val = IO.reshape_old(y_val)
test_Y = IO.reshape_old(test_Y)

# train the model using training set, skip this part if already trained
history = parc_model.fit(
    x=[X_train, X_train_init],
    y=[y_train[:, :, :, :38], y_train[:, :, :, 38:]],
    validation_data=([X_val, X_val_init], [y_val[:, :, :, :38], y_val[:, :, :, 38:]]),
    batch_size=1,
    epochs=300,
)

# save trained model weights
parc_model.save_weights("PARC_trained_data.h5")

# load trained model weights
parc_model.load_weights("PARC_trained_data.h5")

# prediction using trained model parameters
pred = parc_model.predict([test_X, test_X_init])

# reshape to match new data format
test_Y = IO.reshape_new(test_Y)
pred = IO.reshape_new(pred[0], 2)

# definition of temperature prediction and ground truth with rescaling
norm_T_min = normalizing_constants.item().get("Temperature").get("min")
norm_T_max = normalizing_constants.item().get("Temperature").get("max")
Temp_pred = pred[:, :, :, :, 0]
Temp_pred = IO.rescale(Temp_pred, norm_T_min, norm_T_max)
Temp_gt = test_Y[:, :, :, :, 0]
Temp_gt = IO.rescale(Temp_gt, norm_T_min, norm_T_max)

# definition of pressure prediction and ground truth with rescaling
norm_P_min = normalizing_constants.item().get("Pressure").get("min")
norm_P_max = normalizing_constants.item().get("Pressure").get("max")
Pres_pred = pred[:, :, :, :, 1]
Pres_pred = IO.rescale(Pres_pred, norm_P_min, norm_P_max)
Pres_gt = test_Y[:, :, :, :, 1]
Pres_gt = IO.rescale(Pres_gt, norm_P_min, norm_P_max)

# calculate root mean squared, r2 score, and losses for temperature
rmse = losses.rmse(Temp_gt, Temp_pred)
r2 = losses.r2(Temp_gt, Temp_pred)
loss_cv = losses.step_weighted_loss(Temp_gt, Temp_pred, weight_loss)
# step_loss = losses.step_weighted_physical_loss(Temp_gt,Temp_pred,loss_cv)
state_loss = losses.state_weighted_loss(Temp_gt, Temp_pred)

# create plots for data
graphs.visualize_inference(Temp_gt, Temp_pred, t_idx, 2, norm_T_min, norm_T_max)
graphs.visualize_inference(Pres_gt, Pres_pred, t_idx, 2, norm_P_min, norm_P_max)
graphs.plot_rmse(rmse, t_idx)
graphs.plot_r2(r2, t_idx)
graphs.plot_hotspot_area(Temp_gt, Temp_pred, t_idx)
graphs.plot_hotspot_area_dot(Temp_gt, Temp_pred, t_idx, del_t)
graphs.plot_hotspot_temperature(Temp_gt, Temp_pred, t_idx)
graphs.plot_hotspot_temp_dot(Temp_gt, Temp_pred, t_idx, del_t)
graphs.plot_saliency(Temp_pred, 5)
