from scipy import stats as st
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


def compute_KLD(y_true, y_pred):
    """compute KL-divergence
    :param y_true: (numpy)
    :param y_pred: (numpy)
    """

    mean_X = np.mean(y_true)
    sigma_X = np.std(y_true)

    mean_Y = np.mean(y_pred)
    sigma_Y = np.std(y_pred)

    v1 = sigma_X * sigma_X
    v2 = sigma_Y * sigma_Y
    a = np.log(sigma_Y / sigma_X)
    num = v1 + (mean_X - mean_Y) ** 2
    den = 2 * v2
    b = num / den
    return a + b - 0.5


def compute_quantitative_evaluation_sensitivity(y_trues, y_preds):
    """ Calculate average rmse, kld, and pearson correlation value 
        across all time step of given sensitivity value derive from
        DNS (y_trues) w.r.t corresponding PARC predicted value (y_preds)

    :param y_trues: DNS ground truth sensitivity value
    :param y_preds: PARC predicted sensitivty value
    :return [0]:    (float) average rsme 
    :return [1]:    (float) average kld 
    :return [2]:    (float) average pearson correlation

    # """

    pcc_list = []
    rmse_list = []
    kld_list = [] 
    ts = y_preds.shape[1]
    for i in range(ts):
        pcc = st.pearsonr(y_trues[:,i],y_preds[:,i])
        temp_rmse =  sqrt(mean_squared_error(y_trues[:,i],y_preds[:,i]))
        kld = compute_KLD(y_trues[:,i],y_preds[:,i])
        pcc_list.append(pcc[0])
        rmse_list.append(temp_rmse)
        kld_list.append(kld)

    return np.mean(rmse_list), np.mean(kld_list), np.mean(pcc_list)