from parc import model
from parc import IO
from parc import losses
import keras


x_tn, y_tn, x_val, y_val, x_tt, y_tt = IO.getData(
    data_dir="path/to/dataset", splits=[0.8, 0.1, 0.1]
)
x_tn_init = None  # update with correct value
x_val_init = None  # update with correct value
parc_model = model.parc()
loss = ["mse", "mse"]
loss_weight = [1, 1]
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
metrics = ["mae"]

history = model.fit(
    x=[x_tn, x_tn_init],
    y=y_tn[:, :, :, :38, y_tn[:, :, :, 38:]],
    validation_data=([x_val, x_val_init], [y_val[:, :, :, :38], y_val[:, :, :, 38:]]),
    batch_size=1,
    epochs=2,
)

# todo: fill in the missings in the main
