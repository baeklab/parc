import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *


def derivative_solver(temperature, features):
    # initialize integral block structure
    deriv_res_block1_conv0 = Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    deriv_res_block1_conv1 = Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    deriv_res_block1_conv2 = Conv2D(
        64, 3, padding="same", kernel_initializer="he_normal", dtype=tf.float32
    )

    deriv_res_block2_conv0 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    deriv_res_block2_conv1 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    deriv_res_block2_conv2 = Conv2D(
        128, 3, padding="same", kernel_initializer="he_normal", dtype=tf.float32
    )

    deriv_res_block3_conv0 = Conv2D(
        128,
        7,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    deriv_res_block3_conv1 = Conv2D(
        64,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    deriv_res_block3_conv2 = Conv2D(
        32,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )

    deriv_T_dot = Conv2D(
        2,
        3,
        activation="tanh",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )

    # combine temperature and feature map
    concat = keras.layers.concatenate([temperature, features], axis=3)

    # ResNet block #1
    b1_conv0 = deriv_res_block1_conv0(concat)
    b1_conv1 = deriv_res_block1_conv1(b1_conv0)
    b1_conv2 = deriv_res_block1_conv2(b1_conv1)
    b1_add = ReLU()(Add()([b1_conv0, b1_conv2]))

    # ResNet block #2
    b2_conv0 = deriv_res_block2_conv0(b1_add)
    b2_conv1 = deriv_res_block2_conv1(b2_conv0)
    b2_conv2 = deriv_res_block2_conv2(b2_conv1)
    b2_add = ReLU()(Add()([b2_conv0, b2_conv2]))

    # ResNet block #3
    b3_conv0 = deriv_res_block3_conv0(b2_add)
    b3_conv1 = deriv_res_block3_conv1(b3_conv0)
    b3_conv2 = deriv_res_block3_conv2(b3_conv1)
    b3_add = Dropout(0.2)(b3_conv2)

    # output
    Tdot = deriv_T_dot(b3_add)

    return Tdot


def integral_solver(t_dot):
    # initialize integral block structure
    int_res_block1_conv0 = Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    int_res_block1_conv1 = Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    int_res_block1_conv2 = Conv2D(
        64, 3, padding="same", kernel_initializer="he_normal", dtype=tf.float32
    )

    int_res_block2_conv0 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    int_res_block2_conv1 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    int_res_block2_conv2 = Conv2D(
        128, 3, padding="same", kernel_initializer="he_normal", dtype=tf.float32
    )

    int_res_block3_conv0 = Conv2D(
        128,
        7,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    int_res_block3_conv1 = Conv2D(
        64,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )
    int_res_block3_conv2 = Conv2D(
        32,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )

    T_int = Conv2D(
        2,
        3,
        activation="tanh",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )

    # ResNet block #1
    b1_conv0 = int_res_block1_conv0(t_dot)
    b1_conv1 = int_res_block1_conv1(b1_conv0)
    b1_conv2 = int_res_block1_conv2(b1_conv1)
    b1_add = ReLU()(Add()([b1_conv0, b1_conv2]))

    # ResNet block #2
    b2_conv0 = int_res_block2_conv0(b1_add)
    b2_conv1 = int_res_block2_conv1(b2_conv0)
    b2_conv2 = int_res_block2_conv2(b2_conv1)
    b2_add = ReLU()(Add()([b2_conv0, b2_conv2]))
    b2_add = Dropout(0.2)(b2_add)

    # ResNet block #3
    b3_conv0 = int_res_block3_conv0(b2_add)
    b3_conv1 = int_res_block3_conv1(b3_conv0)
    b3_conv2 = int_res_block3_conv2(b3_conv1)
    b3_add = Dropout(0.2)(b3_conv2)

    # output
    Tint = T_int(b3_add)

    return Tint


#
def parc(
    input_size,
    numTS,
    depth,
    kernel_size,
    numFeatureMaps,
):
    # UNet
    inputs = keras.Input(input_size)

    inputs_noise = keras.layers.GaussianNoise(0.01)(inputs)

    conv1 = keras.layers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(inputs_noise)
    conv1 = keras.layers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), dtype=tf.float32)(conv1)

    conv2 = keras.layers.Conv2D(
        128,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(pool1)
    conv2 = keras.layers.Conv2D(
        128,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), dtype=tf.float32)(conv2)

    conv3 = keras.layers.Conv2D(
        256,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(pool2)
    conv3 = keras.layers.Conv2D(
        256,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), dtype=tf.float32)(conv3)

    conv4 = keras.layers.Conv2D(
        512,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(pool3)
    drop4 = keras.layers.Dropout(0.2, dtype=tf.float32)(conv4)

    up6 = keras.layers.UpSampling2D(size=(2, 2), dtype=tf.float32)(drop4)
    merge7 = keras.layers.concatenate([conv3, up6], axis=3)
    conv7 = keras.layers.Conv2D(
        256,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(merge7)
    conv7 = keras.layers.Conv2D(
        256,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(conv7)

    up8 = keras.layers.UpSampling2D(size=(2, 2), dtype=tf.float32)(conv7)
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = keras.layers.Conv2D(
        128,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(up8)
    conv8 = keras.layers.Conv2D(
        128,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(conv8)

    up9 = keras.layers.UpSampling2D(size=(2, 2), dtype=tf.float32)(conv8)
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = keras.layers.Conv2D(
        128,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(merge9)
    feature_map = keras.layers.Conv2D(
        128,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        dtype=tf.float32,
    )(conv9)
    feature_map = keras.layers.Dropout(0.2, dtype=tf.float32)(feature_map)

    # recurrent
    T0_input = keras.Input(input_size)  # todo: seems like not used?

    T0 = keras.Input(input_size)
    T_output_sr = []
    Tdot_output_sr = []
    current_T = T0
    for i in range(numTS - 1):
        current_Tdot = derivative_solver(current_T, feature_map)
        current_T_int = integral_solver(current_Tdot)
        next_T = keras.layers.add([current_T, current_T_int])
        T_output_sr.append(next_T)
        Tdot_output_sr.append(current_Tdot)
        current_T = next_T
        # if i < (numTS-2):
        #    T_output = keras.layers.concatenate([T_output_sr[i+1]],axis=3)
        #    Tdot_output = keras.layers.concatenate([Tdot_output_sr[i+1]],axis=3)

    # todo: below seems to be hardcoded for the timestep, does it adapt to the different timestep?
    T_output = keras.layers.concatenate(
        [
            T_output_sr[0],
            T_output_sr[1],
            T_output_sr[2],
            T_output_sr[3],
            T_output_sr[4],
            T_output_sr[5],
            T_output_sr[6],
            T_output_sr[7],
            T_output_sr[8],
            T_output_sr[9],
            T_output_sr[10],
            T_output_sr[11],
            T_output_sr[12],
            T_output_sr[13],
            T_output_sr[14],
            T_output_sr[15],
            T_output_sr[16],
            T_output_sr[17],
            T_output_sr[18],
        ],
        axis=3,
    )

    # todo: below seems to be hardcoded for the timestep, does it adapt to the different timestep?
    Tdot_output = keras.layers.concatenate(
        [
            Tdot_output_sr[0],
            Tdot_output_sr[1],
            Tdot_output_sr[2],
            Tdot_output_sr[3],
            Tdot_output_sr[4],
            Tdot_output_sr[5],
            Tdot_output_sr[6],
            Tdot_output_sr[7],
            Tdot_output_sr[8],
            Tdot_output_sr[9],
            Tdot_output_sr[10],
            Tdot_output_sr[11],
            Tdot_output_sr[12],
            Tdot_output_sr[13],
            Tdot_output_sr[14],
            Tdot_output_sr[15],
            Tdot_output_sr[16],
            Tdot_output_sr[17],
            Tdot_output_sr[18],
        ],
        axis=3,
    )

    model = keras.Model(inputs=[inputs, T0], outputs=[T_output, Tdot_output])

    return model
