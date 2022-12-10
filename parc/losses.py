import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import keras.backend as K


def rmse(y_true, y_pred): 
    """Root mean squared error calculation between true and predicted cases
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        norm_max (int): maximum value of unscaled data
        norm_min (int): minimum value of unscaled data
    """
    all_rmse = []
    for i in range(y_true.shape[0]):
        rmse_list = []
        for j in range(y_true.shape[3]):
            rmse = sqrt(
                mean_squared_error(
                    y_true[i, :, :, j].flatten(), y_pred[i, :, :, j].flatten()
                )
            )
            rmse_list.append(rmse)

        all_rmse.append(np.array(rmse_list))
    all_rmse = np.array(all_rmse)

    return all_rmse


def r2(y_true, y_pred):
    """R2 score calculation between true and predicted cases
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        case_numbers (int) : number of cases to calculate r2 scores for
        time_steps (int)   : number of time steps to iterate through
    """
    all_r2 = []
    for i in range(y_true.shape[0]):
        r2_list = []
        for j in range(y_true.shape[3]):
            r2 = r2_score(y_true[i, :, :, j].flatten(), y_pred[i, :, :, j].flatten())
            r2_list.append(r2)
        all_r2.append(np.array(r2_list))
    all_r2 = np.array(all_r2)

    return all_r2


def step_weighted_loss(y_true, y_pred, weight_loss):
    """Weighted loss between true and predicted cases, weights for first six time steps are weighted
       with first dimension of weight_loss, the next five with the second dimension, and the final ones
       with the third dimension
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        weight_loss (list[int]): initial weights, middle weights, late weights
    """
    # State loss
    _squared_diff_init = weight_loss[0] * tf.square(
        y_true[:, :, :, :12] - y_pred[:, :, :, :12]
    )
    squared_diff_init = tf.reduce_mean(_squared_diff_init, axis=3)

    _squared_diff_mid = weight_loss[1] * tf.square(
        y_true[:, :, :, :12:22] - y_pred[:, :, :, 12:22]
    )
    squared_diff_mid = tf.reduce_mean(_squared_diff_mid, axis=3)

    _squared_diff_late = weight_loss[2] * tf.square(
        y_true[:, :, :, 22:] - y_pred[:, :, :, 22:]
    )
    squared_diff_late = tf.reduce_mean(_squared_diff_late, axis=3)
    squared_sum = squared_diff_init + squared_diff_mid + squared_diff_late

    loss_cv = tf.reduce_mean(squared_sum, axis=1)
    loss_cv = tf.reduce_mean(loss_cv, axis=1)

    return loss_cv


def step_weighted_physical_loss(y_true, y_pred, loss_cv):
    """calculates physical loss using weighted steps
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
        loss_cv (tf.tensor): state loss
    """
    # Physical loss
    t_true = y_true[:, :, :, 0::2]
    t_mask_true = t_true > -0.6

    t_mask_true = tf.cast(t_mask_true, tf.float32)
    area_true = K.cast(tf.math.count_nonzero(t_mask_true), tf.float32)

    t_true = tf.math.multiply(t_true, t_mask_true)
    ave_ht_true = K.sum(t_true, axis=1)
    ave_ht_true = K.sum(ave_ht_true, axis=1)

    ave_ht_true = ave_ht_true / area_true

    t_pred = y_pred[:, :, :, 0::2]
    t_mask_pred = t_pred > -0.6
    t_mask_pred = tf.cast(t_mask_pred, tf.float32)
    area_pred = K.cast(tf.math.count_nonzero(t_mask_pred), tf.float32)

    t_pred = tf.math.multiply(t_pred, t_mask_pred)
    ave_ht_pred = K.sum(t_pred, axis=1)
    ave_ht_pred = K.sum(ave_ht_pred, axis=1)
    ave_ht_pred = ave_ht_pred / area_pred

    loss_ht = K.square(ave_ht_pred - ave_ht_true)

    loss = loss_cv + 5 * loss_ht

    return loss


def state_weighted_loss(y_true, y_pred):
    """calculates weighted loss for temperature and pressure states
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
    """
    # Temperature loss
    t_pred = y_pred[:, :, :, 0::2]
    t_true = y_true[:, :, :, 0::2]
    mse_temp = tf.reduce_mean(tf.square(t_true - t_pred), axis=3)
    print("temp loss: ", mse_temp)

    # Pressure loss
    p_pred = y_pred[:, :, :, 1::2]
    p_true = y_true[:, :, :, 1::2]
    mse_press = 10 * tf.reduce_mean(tf.square(p_true - p_pred), axis=3)
    print("pressure loss: ", mse_press)

    # Final loss
    squared_sum = mse_temp + mse_press

    loss = tf.reduce_mean(squared_sum, axis=1)
    loss = tf.reduce_mean(squared_sum, axis=1)
    return loss


def sensitivity_single_sample(test_data):
    """single sample sensitivity calculation
    Args:
        test_data (np.ndarray): prediction temp/press values to test sensitivity
        norm_max (int): maximum value of unscaled data
        norm_min (int): minimum value of unscaled data
    """
    threshold = 875 #875 Temperature (K)
    area_list = []
    temp_list = []

    # Calculate area and avg hotspot
    for i in range(test_data.shape[2]):
        pred_slice = test_data[:, :, i]
        pred_mask = pred_slice > threshold
        pred_hotspot_area = np.count_nonzero(pred_mask)
        rescaled_area = pred_hotspot_area * ((2*25/485)**2)
        area_list.append(rescaled_area)
        masked_pred = pred_slice*pred_mask
    
        if pred_hotspot_area ==0:
            pred_avg_temp=0.0
        else:
            pred_avg_temp = np.sum(masked_pred)/pred_hotspot_area
        temp_list.append(pred_avg_temp)
        
    return area_list, temp_list


def calculate_avg_sensitivity(y_true, y_pred):
    """average sensitivity calculation between prediction and true values
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
    """
    whole_temp = []
    whole_area = []
    for i in range(y_true.shape[0]):
        area_pred_list,temp_pred_list = sensitivity_single_sample(y_pred[i, :, :, :])
        whole_temp.append(temp_pred_list)
        whole_area.append(area_pred_list)

    whole_temp = np.array(whole_temp)
    whole_area = np.array(whole_area)

    temp_mean = np.mean(whole_temp,axis=0)
    area_mean = np.mean(whole_area,axis=0)

    temp_error1 = np.percentile(whole_temp,95,axis=0)
    temp_error2 = np.percentile(whole_temp,5,axis=0)
    area_error1 =np.percentile(whole_area,95,axis=0)
    area_error2 =np.percentile(whole_area,5,axis=0)

    gt_whole_temp = []
    gt_whole_area = []
    
    for i in range(y_true.shape[0]):
        area_gt_list,temp_gt_list = sensitivity_single_sample(y_true[i, :, :, :])
        gt_whole_temp.append(temp_gt_list)
        gt_whole_area.append(area_gt_list)
    gt_whole_temp = np.array(gt_whole_temp)
    gt_whole_area = np.array(gt_whole_area)

    gt_temp_mean = np.mean(gt_whole_temp,axis=0)
    gt_area_mean = np.mean(gt_whole_area,axis=0)
    
    gt_temp_error1 = np.percentile(gt_whole_temp,95,axis=0)
    gt_temp_error2 = np.percentile(gt_whole_temp,5,axis=0)
    
    gt_area_error1 =np.percentile(gt_whole_area,95,axis=0)
    gt_area_error2 =np.percentile(gt_whole_area,5,axis=0)

    return (
        temp_mean,
        area_mean,
        temp_error1,
        temp_error2,
        area_error1,
        area_error2,
        gt_temp_mean,
        gt_area_mean,
        gt_temp_error1,
        gt_temp_error2,
        gt_area_error1,
        gt_area_error2
    )