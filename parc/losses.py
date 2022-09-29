import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import keras.backend as K

def calculate_rmse(y_true, y_pred):
    all_rmse = []
    for i in range(9):
        rmse_list = []
        for j in range(19):
            rmse = sqrt(mean_squared_error(y_true[i,:,:,j].flatten(), y_pred[i,:,:,j].flatten()))        
            rmse_list.append(rmse)
        
        all_rmse.append(np.array(rmse_list))    
    all_rmse = np.array(all_rmse)
    
    return all_rmse

def calculate_r2(y_true, y_pred):
    all_r2 = []
    for i in range(9):
        r2_list = []
        for j in range(19):
            r2 = r2_score(y_true[i,:,:,j].flatten(),y_pred[i,:,:,j].flatten())
            r2_list.append(r2)
        all_r2.append(np.array(r2_list))
    all_r2 = np.array(all_r2)
    
    return all_r2

def step_weighted_loss(y_true, y_pred):
    # State loss
    _squared_diff_init = 4*tf.square(y_true[:,:,:,:12] - y_pred[:,:,:,:12])
    squared_diff_init = tf.reduce_mean(_squared_diff_init,axis = 3)

    _squared_diff_mid = 1*tf.square(y_true[:,:,:,:12:22] - y_pred[:,:,:,12:22])
    squared_diff_mid = tf.reduce_mean(_squared_diff_mid,axis = 3)

    _squared_diff_late = tf.square(y_true[:,:,:,22:] -  y_pred[:,:,:,22:])
    squared_diff_late = tf.reduce_mean(_squared_diff_late,axis = 3)
    squared_sum = (squared_diff_init + squared_diff_mid + squared_diff_late)

    loss_cv = tf.reduce_mean(squared_sum, axis = 1)
    loss_cv = tf.reduce_mean(loss_cv, axis = 1)
    
    return loss_cv

def step_weighted_physical_loss(y_true, y_pred):
    # State loss
    _squared_diff_init = 12*tf.square(y_true[:,:,:,:12] - y_pred[:,:,:,:12])
    squared_diff_init = tf.reduce_mean(_squared_diff_init,axis = 3)

    _squared_diff_mid = 2*tf.square(y_true[:,:,:,:12:24] - y_pred[:,:,:,12:24])
    squared_diff_mid = tf.reduce_mean(_squared_diff_init,axis = 3)

    _squared_diff_late = tf.square(y_true[:,:,:,24:] -  y_pred[:,:,:,24:])
    squared_diff_late = tf.reduce_mean(_squared_diff_late,axis = 3)
    squared_sum = (squared_diff_init + squared_diff_mid + squared_diff_late)

    loss_cv = tf.reduce_mean(squared_sum, axis = 1)
    loss_cv = tf.reduce_mean(loss_cv, axis = 1)
    
    # Physical loss
    t_true = y_true[:,:,:,0::2]
    t_mask_true = t_true > -0.6

    t_mask_true = tf.cast(t_mask_true, tf.float32)
    area_true = K.cast(tf.math.count_nonzero(t_mask_true), tf.float32)

    t_true = tf.math.multiply(t_true,t_mask_true)
    ave_ht_true = K.sum(t_true, axis = 1)
    ave_ht_true = K.sum(ave_ht_true, axis = 1)
    
    ave_ht_true = ave_ht_true/area_true
    
    t_pred = y_pred[:,:,:,0::2]
    t_mask_pred = t_pred > -0.6
    t_mask_pred = tf.cast(t_mask_pred, tf.float32)
    area_pred = K.cast(tf.math.count_nonzero(t_mask_pred), tf.float32)
    
    t_pred = tf.math.multiply(t_pred,t_mask_pred)
    ave_ht_pred = K.sum(t_pred, axis = 1)
    ave_ht_pred = K.sum(ave_ht_pred, axis = 1)
    ave_ht_pred = ave_ht_pred/area_pred
    
    loss_ht =  K.square(ave_ht_pred - ave_ht_true)

    loss = loss_cv + 5*loss_ht
    
    return loss

def state_weighted_loss(y_true, y_pred):
  # Temperature loss
  t_pred = y_pred[:,:,:,0::2]
  t_true = y_true[:,:,:,0::2]
  mse_temp =  tf.reduce_mean(tf.square(t_true - t_pred),axis = 3)
  print('temp loss: ', mse_temp)

  # Pressure loss
  p_pred = y_pred[:,:,:,1::2]
  p_true = y_true[:,:,:,1::2]
  mse_press =  10*tf.reduce_mean(tf.square(p_true - p_pred),axis = 3)
  print('pressure loss: ', mse_press)
  # Final loss
  squared_sum = (mse_temp + mse_press)

  loss = tf.reduce_mean(squared_sum, axis = 1)
  loss = tf.reduce_mean(squared_sum, axis = 1)
  return loss