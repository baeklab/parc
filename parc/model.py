import tensorflow as tf
import keras


class PARC(keras.Model):
    def __init__(
        self, input_size, n_fields=2, n_timesteps=19, n_featuremaps=128, **kwargs
    ):
        super(PARC, self).__init__(**kwargs)
        self.n_fields = n_fields
        self.n_timesteps = n_timesteps
        self.n_featuremaps = n_featuremaps
        self.input_size = input_size

        # derivitave solver initialization
        self.deriv_res_block1_conv0 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_1_0",
        )
        self.deriv_res_block1_conv1 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_1_1",
        )
        self.deriv_res_block1_conv2 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            3,
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_1_2",
        )

        self.deriv_res_block2_conv0 = keras.layers.Conv2D(
            self.n_featuremaps,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_2_0",
        )
        self.deriv_res_block2_conv1 = keras.layers.Conv2D(
            self.n_featuremaps,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_2_1",
        )
        self.deriv_res_block2_conv2 = keras.layers.Conv2D(
            self.n_featuremaps,
            3,
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_2_2",
        )

        self.deriv_res_block3_conv0 = keras.layers.Conv2D(
            self.n_featuremaps,
            7,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_3_0",
        )
        self.deriv_res_block3_conv1 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_3_1",
        )
        self.deriv_res_block3_conv2 = keras.layers.Conv2D(
            self.n_featuremaps / 4,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_3_2",
        )

        self.deriv_F_dot = keras.layers.Conv2D(
            self.n_fields,
            3,
            activation="tanh",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="deriv_out",
        )

        # initialize integral block structure
        self.int_res_block1_conv0 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_1_0",
        )
        self.int_res_block1_conv1 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_1_1",
        )
        self.int_res_block1_conv2 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            3,
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_1_2",
        )

        self.int_res_block2_conv0 = keras.layers.Conv2D(
            self.n_featuremaps,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_2_0",
        )
        self.int_res_block2_conv1 = keras.layers.Conv2D(
            self.n_featuremaps,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_2_1",
        )
        self.int_res_block2_conv2 = keras.layers.Conv2D(
            self.n_featuremaps,
            3,
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_2_2",
        )

        self.int_res_block3_conv0 = keras.layers.Conv2D(
            self.n_featuremaps,
            7,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_3_0",
        )
        self.int_res_block3_conv1 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_3_1",
        )
        self.int_res_block3_conv2 = keras.layers.Conv2D(
            self.n_featuremaps / 4,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_3_2",
        )

        self.F_int = keras.layers.Conv2D(
            self.n_fields,
            3,
            activation="tanh",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="int_out",
        )

        # shape descriptors (UNet)
        self.conv1_1 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_1_1",
        )
        self.conv1_2 = keras.layers.Conv2D(
            self.n_featuremaps / 2,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_1_2",
        )

        self.conv2_1 = keras.layers.Conv2D(
            self.n_featuremaps,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_2_1",
        )
        self.conv2_2 = keras.layers.Conv2D(
            self.n_featuremaps,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_2_2",
        )

        self.conv3_1 = keras.layers.Conv2D(
            self.n_featuremaps * 2,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_3_1",
        )
        self.conv3_2 = keras.layers.Conv2D(
            self.n_featuremaps * 2,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_3_2",
        )

        self.conv4 = keras.layers.Conv2D(
            self.n_featuremaps * 4,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_down_4_1",
        )

        self.conv7_1 = keras.layers.Conv2D(
            self.n_featuremaps * 2,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_up_7_1",
        )
        self.conv7_2 = keras.layers.Conv2D(
            self.n_featuremaps * 2,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_up_7_2",
        )

        self.conv8_1 = keras.layers.Conv2D(
            self.n_featuremaps,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_up_8_1",
        )
        self.conv8_2 = keras.layers.Conv2D(
            self.n_featuremaps,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_up_8_2",
        )

        self.conv9_1 = keras.layers.Conv2D(
            self.n_featuremaps,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_up_9_1",
        )
        self.feature_map = keras.layers.Conv2D(
            self.n_featuremaps,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
            name="unet_out",
        )

        self.input_layer1 = keras.layers.Input((self.input_size, self.input_size, 2))
        self.input_layer2 = keras.layers.Input(
            (self.input_size, self.input_size, self.n_fields)
        )
        self.out = self.call([self.input_layer1, self.input_layer2])

        super(PARC, self).__init__(
            inputs=[self.input_layer1, self.input_layer2], outputs=self.out, **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=[self.input_layer1, self.input_layer2], outputs=self.out
        )

    def call(self, inputs, training=True):
        microstructure = inputs[0]
        F_initial = inputs[1]

        # shape descriptor: UNet
        x = keras.layers.GaussianNoise(0.01)(microstructure)

        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv4(pool3)
        drop4 = keras.layers.Dropout(0.2)(conv4)

        up6 = keras.layers.UpSampling2D(size=(2, 2))(drop4)
        merge7 = keras.layers.concatenate([conv3, up6], axis=3)
        conv7 = self.conv7_1(merge7)
        conv7 = self.conv7_2(conv7)

        up8 = keras.layers.UpSampling2D(size=(2, 2))(conv7)

        conv8 = self.conv8_1(up8)  # Scientific Advance version:
        conv8 = self.conv8_2(conv8)

        up9 = keras.layers.UpSampling2D(size=(2, 2))(conv8)
        merge9 = keras.layers.concatenate([conv1, up9], axis=3)

        conv9 = self.conv9_1(merge9)
        feature_map = self.feature_map(conv9)
        feature_map = keras.layers.Dropout(0.2)(feature_map)

        # Recurrent formulation
        F_dots, Fs = [], []
        F_current = F_initial
        for i in range(self.n_timesteps):
            Fdot_i = self.derivative_solver(F_current, feature_map)
            Fint_i = self.integral_solver(Fdot_i)
            F_current = keras.layers.add(
                [F_current, Fint_i]
            )  # update for next timestep

            Fs.append(F_current)
            F_dots.append(Fdot_i)

        F_output = keras.layers.concatenate(Fs, axis=3)
        Fdot_output = keras.layers.concatenate(F_dots, axis=3)

        return F_output, Fdot_output

    def derivative_solver(self, Fs, features):
        concat = keras.layers.concatenate([Fs, features], axis=3)

        # ResNet block #1
        b1_conv0 = self.deriv_res_block1_conv0(concat)
        b1_conv1 = self.deriv_res_block1_conv1(b1_conv0)
        b1_conv2 = self.deriv_res_block1_conv2(b1_conv1)
        b1_add = keras.layers.ReLU()(keras.layers.Add()([b1_conv0, b1_conv2]))

        # ResNet block #2
        b2_conv0 = self.deriv_res_block2_conv0(b1_add)
        b2_conv1 = self.deriv_res_block2_conv1(b2_conv0)
        b2_conv2 = self.deriv_res_block2_conv2(b2_conv1)
        b2_add = keras.layers.ReLU()(keras.layers.Add()([b2_conv0, b2_conv2]))

        # ResNet block #3
        b3_conv0 = self.deriv_res_block3_conv0(b2_add)
        b3_conv1 = self.deriv_res_block3_conv1(b3_conv0)
        b3_conv2 = self.deriv_res_block3_conv2(b3_conv1)
        b3_add = keras.layers.Dropout(0.2)(b3_conv2)

        # output
        Fdot = self.deriv_F_dot(b3_add)
        return Fdot

    def integral_solver(self, t_dot):
        # ResNet block #1
        b1_conv0 = self.int_res_block1_conv0(t_dot)
        b1_conv1 = self.int_res_block1_conv1(b1_conv0)
        b1_conv2 = self.int_res_block1_conv2(b1_conv1)
        b1_add = keras.layers.ReLU()(keras.layers.Add()([b1_conv0, b1_conv2]))

        # ResNet block #2
        b2_conv0 = self.int_res_block2_conv0(b1_add)
        b2_conv1 = self.int_res_block2_conv1(b2_conv0)
        b2_conv2 = self.int_res_block2_conv2(b2_conv1)
        b2_add = keras.layers.ReLU()(keras.layers.Add()([b2_conv0, b2_conv2]))
        b2_add = keras.layers.Dropout(0.2)(b2_add)

        # ResNet block #3
        b3_conv0 = self.int_res_block3_conv0(b2_add)
        b3_conv1 = self.int_res_block3_conv1(b3_conv0)
        b3_conv2 = self.int_res_block3_conv2(b3_conv1)
        b3_add = keras.layers.Dropout(0.2)(b3_conv2)

        # output
        Fint = self.F_int(b3_add)

        return Fint