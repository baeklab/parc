import tensorflow as tf
import keras
from keras.layers import *


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
    
def resnet_unit(feat_dim, kernel_size, x_in):
    # conv = Conv2D(feats, kernel, padding="same")
    res = keras.Sequential([
        Conv2D(feat_dim, kernel_size, padding = "same"
               ,kernel_initializer = 'he_uniform'
               ,bias_initializer = 'he_uniform',
               kernel_regularizer = regularizers.L1(l1=1e-3)
              ),
        ReLU(),
        Conv2D(feat_dim, kernel_size, padding = "same", 
               kernel_initializer = 'he_uniform',
               bias_initializer = 'he_uniform',
               kernel_regularizer = regularizers.L1(l1=1e-3))
    ])
    return ReLU()(x_in + res(x_in))

def resnet_block(feat_dim, reps, x_in, pooling = True):
    # Stage 2
    conv1 = Conv2D(feat_dim, 3, padding = "same", activation = 'relu', 
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-3))(x_in)
    x = Conv2D(feat_dim, 3, padding = "same", activation = 'relu', 
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-3))(conv1)
    for _ in range(reps):
        x = resnet_unit(feat_dim,3,x)
    if pooling == True:
        x = MaxPooling2D(2,2)(x)
        return x
    else:
        return x

class SPADE(keras.layers.Layer):
    def __init__(self, filters, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = keras.layers.Conv2D(filters, 3, padding="same", activation="relu",
                                  kernel_initializer='he_uniform', bias_initializer='he_uniform',
                                  kernel_regularizer=regularizers.L1(l1=1e-3))
        self.conv_gamma = keras.layers.Conv2D(filters, 3, padding="same",
                                        kernel_initializer='he_uniform', bias_initializer='he_uniform',
                                        kernel_regularizer=regularizers.L1(l1=1e-3))
        self.conv_beta = keras.layers.Conv2D(filters, 3, padding="same",
                                       kernel_initializer='he_uniform', bias_initializer='he_uniform',
                                       kernel_regularizer=regularizers.L1(l1=1e-3))

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        output = gamma * normalized + beta
        return output

def spade_generator_unit(feats_in, feats_out, kernel, x, mask, upsampling = True):
    x = GaussianNoise(0.05)(x)
    # SPADE & conv
    spade1 = SPADE(feats_in)(x, mask)
    conv1 = Conv2D(feats_in,kernel, padding='same', activation= LeakyReLU(0.2),
                   kernel_initializer='he_uniform', bias_initializer='he_uniform',
                   kernel_regularizer=regularizers.L1(l1=1e-3))(spade1)
   
    conv_out = Conv2D(feats_out,kernel, padding='same', 
                      kernel_initializer='he_uniform', bias_initializer='he_uniform',
                      kernel_regularizer=regularizers.L1(l1=1e-3))(conv1)
    output = LeakyReLU(0.2)(conv_out)
    if upsampling == True:
        output = UpSampling2D(size = (2,2))(output)
        return output
    else:
        return output

def resnet_based_encoder(latent_dims = 512, input_shape = (224,448,3)):
    inputs = keras.Input(shape = input_shape)
    x = Conv2D(64, 7, padding="same", activation = 'relu', 
               kernel_initializer='he_uniform', bias_initializer='he_uniform', 
               kernel_regularizer=regularizers.L1(l1=1e-3))(inputs)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.3)(x)
    x1 = resnet_block(64,1,x,pooling = True)
    x1 = Dropout(0.3)(x1)
    x2 = resnet_block(128,2,x1,pooling = True)
    x2 = Dropout(0.3)(x2)
    x3 = resnet_block(256,2,x2,pooling = True)
    x3 = Dropout(0.3)(x3)
    
    x_out = Conv2D(64, 1, padding = "same", 
                kernel_initializer = 'he_uniform', bias_initializer='he_uniform',
                kernel_regularizer = regularizers.L1(l1=1e-3),
                activity_regularizer = regularizers.L1(1e-3))(x3)
    encoder = keras.Model(inputs, x_out)
    return encoder
# resnet_based_encoder().summary()

def spade_based_generator(output_activation = 'linear'):
    latent_input = keras.Input(shape = (14,28,64))
    latent_input = GaussianNoise(0.1)(latent_input)
    sp_bl_init = Conv2D(256, 3, padding = "same", activation = LeakyReLU(0.2),
                        kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                        kernel_regularizer = regularizers.L1(l1=1e-3))(latent_input)
    sp_bl_init = Dropout(0.3)(sp_bl_init)
    sp_bl2 = spade_generator_unit(256,128,3,sp_bl_init,latent_input, upsampling = True)
    sp_bl2 = Dropout(0.3)(sp_bl2)

    sp_bl3 = spade_generator_unit(128,64,3,sp_bl2,latent_input, upsampling = True)
    sp_bl3 = Dropout(0.3)(sp_bl3)

    sp_bl4 = spade_generator_unit(64,64,3,sp_bl3,latent_input, upsampling = True)
    sp_bl4 = Dropout(0.3)(sp_bl4)

    sp_bl5 = spade_generator_unit(64,64,3,sp_bl4,latent_input, upsampling = True)
    if output_activation == 'sigmoid':
        recon = Conv2D(3,3,padding = "same",activation = 'sigmoid',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                       kernel_regularizer = regularizers.L1(l1=1e-3))(sp_bl5)
    else:
        recon = Conv2D(3,3,padding = "same",
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                       kernel_regularizer = regularizers.L1(l1=1e-3))(sp_bl5)
    decoder = keras.Model(latent_input, recon)
    return decoder
# spade_based_generator().summary()

def latent_evolution_model():
    latent_input = keras.Input(shape = (14,28,64))
    conv1 = Conv2D(64, 1, padding = 'same', 
                   kernel_initializer='he_uniform', bias_initializer='he_uniform', 
                   kernel_regularizer=regularizers.L1(l1=1e-3))(latent_input)
    conv2 = Conv2D(64, 1, padding = 'same', 
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-3))(conv1)
    conv3 = Conv2D(64, 1, padding = 'same', 
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform', 
                   kernel_regularizer = regularizers.L1(l1=1e-3))(conv2)
    latent_out = keras.Model(latent_input, conv3)
    return latent_out


class PILE(keras.Model):
    def __init__(self, **kwargs):
        super(PILE, self).__init__(**kwargs)
        self.encoder = resnet_based_encoder()
        self.decoder = spade_based_generator()
        self.latent_evolution = latent_evolution_model()
        # loss define
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.in_ae_loss_tracker = keras.metrics.Mean(name="in_ae_loss")
        self.out_ae_loss_tracker = keras.metrics.Mean(name="out_ae_loss")
        self.latent_loss_tracker = keras.metrics.Mean(name="latent_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        self.in_ae_loss_tracker,
        self.out_ae_loss_tracker,
        self.latent_loss_tracker,
        self.recon_loss_tracker
        ]

    def train_step(self, data):

        input_temp = data[0]
        output_temp = data[1]
        total_loss = 0
        with tf.GradientTape() as tape:
            # AE loss
            latent_temp_in = self.encoder(input_temp)
            ae_recon_temp_in = self.decoder(latent_temp_in)
            in_ae_loss = tf.keras.losses.MeanSquaredError()(input_temp,ae_recon_temp_in)
            latent_temp_out = self.encoder(output_temp)
            ae_recon_temp_out = self.decoder(latent_temp_out)
            out_ae_loss = tf.keras.losses.MeanSquaredError()(output_temp,ae_recon_temp_out)

            ae_loss = in_ae_loss + out_ae_loss

            pred_latent_temp_out = self.latent_evolution(latent_temp_in, latent_temp_out)
            latent_loss = tf.keras.losses.MeanSquaredError()(pred_latent_temp_out,latent_temp_out)
            recon_latent_temp = self.decoder(pred_latent_temp_out)
            recon_loss = tf.keras.losses.MeanSquaredError()(output_temp,recon_latent_temp)
            

            total_loss =  recon_loss + ae_loss + latent_loss
            
        self.total_loss_tracker.update_state(total_loss)
        self.in_ae_loss_tracker.update_state(in_ae_loss)
        self.out_ae_loss_tracker.update_state(out_ae_loss)
        self.latent_loss_tracker.update_state(latent_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))


        return {
            "total_loss": self.total_loss_tracker.result(),
            "in_ae_loss": self.in_ae_loss_tracker.result(),
            "out_ae_loss": self.out_ae_loss_tracker.result(),
            "latent_loss": self.latent_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
        }
    
    
class PILR(keras.Model):
    def __init__(self, **kwargs):
        super(PILR, self).__init__(**kwargs)
        self.encoder = resnet_based_encoder()
        self.decoder = spade_based_generator(output_activation = 'sigmoid')

        # loss define
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def train_step(self, data):
        total_loss = 0
        with tf.GradientTape() as tape:
            # AE loss
            latent_micro = self.encoder(data)
            recon_micro = self.decoder(latent_micro)
            total_loss = tf.keras.losses.BinaryCrossentropy()(data,recon_micro)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }

def latent_physics():
    #z_m = keras.Input(shape = (14,28,64))
    z_t = keras.Input(shape = (14,28,64))
    
#     concat = Concatenate()([z_m,z_t])
    #z_m_noise = GaussianNoise(0.05)(z_m)  
    z_t_noise = GaussianNoise(0.05)(z_t)  

    # spade block
    sp_bl = Dropout(0.2)(z_t_noise)

    # inception block 1
    conv11 = Conv2D(128,1, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(sp_bl)
    
    conv21 = Conv2D(128,1, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(sp_bl)
    conv21 = Conv2D(128,3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv21)
    
    conv31 = Conv2D(128,1, activation = 'relu', padding = 'same', 
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(sp_bl)
    conv31 = Conv2D(128,5, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv31)
    
    incept1 = Concatenate()([Concatenate()([conv11,conv21]),conv31])
    incept1 = Dropout(0.2)(incept1)

    # Nonlinear mapping
    conv_out1 = Conv2D(256,1, activation = 'relu', padding = 'same',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(incept1)
    conv_out1 = Dropout(0.2)(conv_out1)
    
    conv_out2 = Conv2D(512,1, activation = 'relu', padding = 'same',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv_out1)    
    conv_out2 = Dropout(0.2)(conv_out2)

    conv_out3 = Conv2D(1024,1, activation = 'relu', padding = 'same',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv_out2)
    conv_out3 = Dropout(0.2)(conv_out3)

    conv_out = Conv2D(64, 1, padding = 'same', 
                      kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv_out3)

    #out = z_t + conv_out
  
    latent_out = keras.Model([z_t], conv_out)
    return latent_out


class LatentPARC(keras.Model):
    def __init__(self, ts, **kwargs):
        super(LatentPARC, self).__init__(**kwargs)
        
        self.latent_physics = latent_physics()
        # loss define
        self.mapping = PILE()
        self.mapping.encoder.load_weights('temp_enc_28_56_linear_constraint_v2.h5')
        self.mapping.encoder.trainable = False
        self.mapping.decoder.load_weights('temp_dec_28_56_linear_constraint_v2.h5')
        self.mapping.decoder.trainable = False
        self.mapping.latent_evolution.load_weights('temp_pile_28_56_linear_constraint_v2.h5')
        self.mapping.latent_evolution.trainable = False
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")      
        self.ts = ts

        self.latent_loss_tracker = keras.metrics.Mean(name="latent_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.latent_loss_tracker,
            self.recon_loss_tracker
        ]

    def train_step(self, data):
        init_data = data[0][0]
        data_gt = data[1][0]
        temp = data[1][1]
        latent_loss = 0
        recon_loss = 0
        with tf.GradientTape() as tape:
            latent_current = init_data
            for i in range(self.ts):
                latent_next = self.latent_physics([latent_current])
                latent_current = latent_next
                iter_latent_loss = tf.keras.losses.MeanSquaredError()(latent_next,data_gt[:,:,:,:,i])
                recon = self.mapping.decoder(latent_next)
                iter_recon_loss = tf.keras.losses.MeanSquaredError()(recon,temp[:,:,:,i,:])
                
                latent_loss += iter_latent_loss
                recon_loss += iter_recon_loss

            total_loss = 0.7*recon_loss + 0.3*latent_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.latent_loss_tracker.update_state(latent_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "latent_loss": self.latent_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
        }
