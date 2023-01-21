import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import keras.backend as K
from parc import IO

def rmse(y_true: np.ndarray, y_pred: np.ndarray): 
    """Root mean squared error calculation between true and predicted cases
    Args:
        y_true (np.ndarray): true values for temp or pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp or pressure
    Returns:
        all_rmse (np.ndarray): rmse values for each time step
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


def r2(y_true: np.ndarray, y_pred: np.ndarray):
    """R2 score calculation between true and predicted cases
    Args:
        y_true (np.ndarray): true values for temp or pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp or pressure
    Returns:
        all_r2 (np.ndarray): r2 values for each time step
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


def step_weighted_loss(y_true: np.ndarray, y_pred: np.ndarray, weight_loss: list):
    """Weighted loss between true and predicted cases, weights for first six time steps are weighted
       with first dimension of weight_loss, the next five with the second dimension, and the final ones
       with the third dimension
    Args:
        y_true (np.ndarray): true values for temp or pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp or pressure
        weight_loss (list[int]): initial weights, middle weights, late weights
    Returns:
        loss_cv (tf.Tensor): time-weighted losses for each case
    """
    _squared_diff_init = weight_loss[0] * tf.square(
        y_true[:, :, :, :6] - y_pred[:, :, :, :6]
    )
    squared_diff_init = tf.reduce_mean(_squared_diff_init, axis=3)
    
    _squared_diff_mid = weight_loss[1] * tf.square(
        y_true[:, :, :, :6:11] - y_pred[:, :, :, 6:11]
    )
    squared_diff_mid = tf.reduce_mean(_squared_diff_mid, axis=3)

    _squared_diff_late = weight_loss[2] * tf.square(
        y_true[:, :, :, 11:] - y_pred[:, :, :, 11:]
    )
    squared_diff_late = tf.reduce_mean(_squared_diff_late, axis=3)
       
    squared_sum = squared_diff_init + squared_diff_mid + squared_diff_late

    loss_cv = tf.reduce_mean(squared_sum, axis=1)
    loss_cv = tf.reduce_mean(loss_cv, axis=1)
    
    return loss_cv


def step_weighted_physical_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_cv: tf.Tensor):
    """calculates physical loss using weighted steps
    Args:
        y_true (np.ndarray): true values for temp or pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp or pressure 
        loss_cv (tf.tensor): state loss
    Returns:
        loss (tf.Tensor): total loss for each case
    """
    # Physical loss
    t_true = y_true
    t_mask_true = t_true > -0.6

    t_mask_true = tf.cast(t_mask_true, tf.float64)
    area_true = K.cast(tf.math.count_nonzero(t_mask_true), tf.float64)

    t_true = tf.math.multiply(t_true, t_mask_true)
    ave_ht_true = K.sum(t_true, axis=1)
    ave_ht_true = K.sum(ave_ht_true, axis=1)

    ave_ht_true = ave_ht_true / area_true

    t_pred = y_pred
    t_mask_pred = t_pred > -0.6
    t_mask_pred = tf.cast(t_mask_pred, tf.float64)
    area_pred = K.cast(tf.math.count_nonzero(t_mask_pred), tf.float64)

    t_pred = tf.math.multiply(t_pred, t_mask_pred)
    ave_ht_pred = K.sum(t_pred, axis=1)
    ave_ht_pred = K.sum(ave_ht_pred, axis=1)
    ave_ht_pred = ave_ht_pred / area_pred

    loss_ht = K.square(ave_ht_pred - ave_ht_true)
    loss_ht = tf.reduce_mean(loss_ht, axis=1)

    loss = loss_cv + 5 * loss_ht
    
    return loss


def state_weighted_loss(y_true: np.ndarray, y_pred: np.ndarray):
    """calculates weighted loss based on temperature and pressure states
    Args:
        y_true (np.ndarray): true values for temp/press found in input dataset
        y_pred (np.ndarray): model predicted values for temp/press
    Returns:
        loss (tf.Tensor): total loss for each case
    """
    # Temperature loss
    t_pred = y_pred[:, :, :, :, 0]
    t_true = y_true[:, :, :, :, 0]
    mse_temp = tf.reduce_mean(tf.square(t_true - t_pred), axis=3)

    # Pressure loss
    p_pred = y_pred[:, :, :, :, 1]
    p_true = y_true[:, :, :, :, 1]
    mse_press = 10 * tf.reduce_mean(tf.square(p_true - p_pred), axis=3)

    # Final loss
    squared_sum = mse_temp + mse_press

    loss = tf.reduce_mean(squared_sum, axis=1)
    loss = tf.reduce_mean(loss, axis=1)
    
    return loss


def sensitivity_single_sample(test_data: np.ndarray, threshold: int):
    """finds the hotspots based on temperature threshold sensitivity and determines area and average temperature of this hotspot
    Args:
        test_data (np.ndarray): prediction temp or pressure values to test sensitivity
        threshold (int): value at which to conclude is a valid hotspot
    Returns:
        area_list (list[int]): total hotspot area for each time step and case
        temp_list (list[int]): average hotspot temp or pressure for each time step and case
    """
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

def mean_percentile(whole_data: np.ndarray):
    """calculates mean of data set and the 95th percentiles for error bar visualization
    Args:
        whole_data (np.ndarray): whole set of data to find mean and upper/lower percentiles of
    Returns:
        mean (np.ndarray): mean of the data set
        error1 (np.ndarray): upper percentile of data
        error2 (np.ndarray): lower percentile of data
    """
    mean = np.mean(whole_data,axis=0)
    error1 = np.percentile(whole_data,95,axis=0)
    error2 = np.percentile(whole_data,5,axis=0)

    return mean, error1, error2

def calculate_avg_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, t_idx: list, threshold: int):
    """calculation of mean hotspot area and temperature with error ranges for prediction and true values
    Args:
        y_true (np.ndarray): true values for temp or pressure found in input dataset
        y_pred (np.ndarray): model predicted values for temp or pressure
        t_idx (list[int]) : time index of the test cases
        threshold (int): value at which to conclude is a valid hotspot
    Returns:
        mean_error (dict): dictionary containing all mean and error values for prediction and ground truth temp/area/gradients
    """
    whole_temp = []
    whole_area = []
    for i in range(y_pred.shape[0]):
        area_pred_list,temp_pred_list = sensitivity_single_sample(y_pred[i, :, :, :],threshold)
        whole_temp.append(temp_pred_list)
        whole_area.append(area_pred_list)

    whole_temp = np.array(whole_temp)
    whole_area = np.array(whole_area)
    
    temp_mean, temp_error1, temp_error2 = mean_percentile(whole_temp)
    area_mean, area_error1, area_error2 = mean_percentile(whole_area)
    
    gt_whole_temp = []
    gt_whole_area = []
    
    for i in range(y_true.shape[0]):
        area_gt_list,temp_gt_list = sensitivity_single_sample(y_true[i, :, :, :],threshold)
        gt_whole_temp.append(temp_gt_list)
        gt_whole_area.append(area_gt_list)
    gt_whole_temp = np.array(gt_whole_temp)
    gt_whole_area = np.array(gt_whole_area)
    
    gt_temp_mean, gt_temp_error1, gt_temp_error2 = mean_percentile(gt_whole_temp)
    gt_area_mean, gt_area_error1, gt_area_error2 = mean_percentile(gt_whole_area)
    
    whole_temp_deriv = IO.calculate_derivative(whole_temp, t_idx)
    whole_area_deriv = IO.calculate_derivative(whole_area, t_idx)
    
    T_dot, temp_error1_deriv, temp_error2_deriv = mean_percentile(whole_temp_deriv)
    A_dot, area_error1_deriv, area_error2_deriv = mean_percentile(whole_area_deriv)

    gt_temp_deriv = IO.calculate_derivative(gt_whole_temp, t_idx)
    gt_area_deriv = IO.calculate_derivative(gt_whole_area, t_idx)
    
    T_dot_gt, gt_temp_error1_deriv, gt_temp_error2_deriv = mean_percentile(gt_temp_deriv)
    A_dot_gt, gt_area_error1_deriv, gt_area_error2_deriv = mean_percentile(gt_area_deriv)
    
    
    #create dictionary for mean and error values
    mean_error = {
        'Prediction' : {
            'Temperature' : {
                'mean' : temp_mean, 
                'error1' : temp_error1,
                'error2' : temp_error2
            }, 
            'Area' : {
                'mean' : area_mean,
                'error1' : area_error1,
                'error2' : area_error2
            },
            'Temperature_gradient' : {
                'mean' : T_dot, 
                'error1' : temp_error1_deriv,
                'error2' : temp_error2_deriv
            },
            'Area_gradient' : {
                'mean' : A_dot, 
                'error1' : area_error1_deriv,
                'error2' : area_error2_deriv
            },
        },
        'Ground_truth' : {
            'Temperature' : {
                'mean' : gt_temp_mean, 
                'error1' : gt_temp_error1,
                'error2' : gt_temp_error2
            }, 
            'Area' : {
                'mean' : gt_area_mean,
                'error1' : gt_area_error1,
                'error2' : gt_area_error2
            },
            'Temperature_gradient' : {
                'mean' : T_dot_gt, 
                'error1' : gt_temp_error1_deriv,
                'error2' : gt_temp_error2_deriv
            },
            'Area_gradient' : {
                'mean' : A_dot_gt, 
                'error1' : gt_area_error1_deriv,
                'error2' : gt_area_error2_deriv
            },
        }
    }
    
    return mean_error