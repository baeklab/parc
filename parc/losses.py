import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score


def calculate_rmse(prediction,groundTruth):
    all_rmse = []
    for i in range(9):
        rmse_list = []
        for j in range(19):
            rmse = sqrt(mean_squared_error(groundTruth[i,:,:,j].flatten(), prediction[i,:,:,j].flatten()))        
            rmse_list.append(rmse)
        
        all_rmse.append(np.array(rmse_list))    
    all_rmse = np.array(all_rmse)
    
    return all_rmse

def calculate_r2(prediction,groundTruth):
    all_r2 = []
    for i in range(9):
        r2_list = []
        for j in range(19):
            r2 = r2_score(groundTruth[i,:,:,j].flatten(),prediction[i,:,:,j].flatten())
            r2_list.append(r2)
        all_r2.append(np.array(r2_list))
    all_r2 = np.array(all_r2)
    
    return all_r2

all_rmse = calculate_rmse(prediction,groundTruth)
all_r2 = calculate_r2(prediction,groundTruth)