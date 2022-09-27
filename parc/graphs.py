import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def visualizeInference(
    data_in: np.ndarray,
    data_out: np.ndarray,
    time_idx: list = None,
    fields: list = None,
):
    """plot the inference results

    Args:
        data_in (np.ndarray):   ground truth label
        data_out (np.ndarray):  model prediction
        time_idx (list[int]):   list of the time index to plot. If None plot all timestep
        fields (list[bool]):    list of selecting which fields to plot. If None, plot all fields

    Returns:
        None
    """


def plot_rmse(Prediction,gradient):
    #calculate RMSE for each sample
    all_rmse = []
    for i in range(9):
        rmse_list = []
        r2_list = []
        for j in range(19):
            rmse = sqrt(mean_squared_error(gradient[i,:,:,j].flatten(), Prediction[i,:,:,j].flatten()))        
            rmse_list.append(rmse)
        
        all_rmse.append(np.array(rmse_list))    
    all_rmse = np.array(all_rmse)

    #mean of the RMSE values
    all_rmse_mean = np.mean(all_rmse,axis=0)

    #total RMSE value
    rmse_all = sqrt(mean_squared_error(gradient[:,:,:,:].flatten(), Prediction[:,:,:,:].flatten()))
    print(rmse_all)

    #Root mean squared error plot
    sample_name = 'RMSE'
    plt.figure(figsize=[17,4])
    x_num = np.array([0.79,1.58, 2.37, 3.16,3.95,4.74,5.53,6.32,7.11,7.9,8.69,9.48,10.27,11.06,11.85,12.64,13.43,14.22,15.01])
    plt.boxplot(all_rmse, whis=[5,95], medianprops = dict(linewidth=0),meanline = True, showmeans = True,showfliers = False,labels=None,positions=x_num)
    for i in range(len(all_rmse)):
        plt.scatter(x_num,all_rmse[i,:],alpha=0.4,color='b')

    #Add labels and title
    plt.title(sample_name)
    plt.xlabel("ns")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    return None


def plot_r2(Prediction,gradient):
    #calculate r^2 for sample
    all_r2 = []
    for i in range(9):
        r2_list = []
        for j in range(19):
            r2 = r2_score(gradient[i,:,:,j].flatten(),Prediction[i,:,:,j].flatten())
            r2_list.append(r2)
        all_r2.append(np.array(r2_list))
    all_r2 = np.array(all_r2)

    #mean of the r^2 values
    all_r2_mean = np.mean(all_r2,axis=0)

    #total r^2 score
    r2_all = r2_score(gradient.flatten(),Prediction.flatten())
    
    #R^2 score plot
    sample_name = 'R2'
    plt.figure(figsize=[17,4])
    x_num = np.array([0.79,1.58, 2.37, 3.16,3.95,4.74,5.53,6.32,7.11,7.9,8.69,9.48,10.27,11.06,11.85,12.64,13.43,14.22,15.01])
    plt.boxplot(all_r2, whis=[5,95], medianprops = dict(linewidth=0),meanline = True, showmeans = True,showfliers = False,positions=x_num)
    for i in range(len(all_r2)):
        plt.scatter(x_num,all_r2[i,:],alpha=0.4,color='b')

    #Add labels and title
    plt.title(sample_name)
    plt.xlabel("ns")
    plt.ylabel("R2")
    plt.axis([0.5, 15.5, 0, 1])
    plt.legend()
    plt.show()

    return None


def plot_sensitivity_area(Prediction,gradient):
    #single sample sensitivity calculation for temperature
    def sensitivity_single_sample(test_data):
        threshold = 0.1554 #875 Temperature(K)

        area_list = []
        #Rescale the output
        test_data = (test_data+1.0)/2.0

        #Calculate area and avg hotspot
        for i in range(3,18):
            pred_slice = test_data[:,:,i]
            pred_mask = pred_slice>threshold
            pred_hotspot_area = np.count_nonzero(pred_mask)
            area_list.append(pred_hotspot_area)

        return area_list

    #average sensitivity calculation for temperature
    def Calculate_avg_sensitivity(pred,test_Y):
        whole_area = []
        for i in range(8):
            area_gt_list = sensitivity_single_sample(pred[i,:,:,:])
            whole_area.append(area_gt_list)
            
        whole_area = np.array(whole_area)

        area_mean = np.mean(whole_area,axis=0)

        area_error1 =np.percentile(whole_area,95,axis=0)
        area_error2 =np.percentile(whole_area,5,axis=0)

        gt_whole_area = []
        
        for i in range(8):
            area_pred_list = sensitivity_single_sample(test_Y[i,:,:,:])
            gt_whole_area.append(area_pred_list)
        gt_whole_area = np.array(gt_whole_area)

        gt_area_mean = np.mean(gt_whole_area,axis=0)
        
        gt_area_error1 =np.percentile(gt_whole_area,95,axis=0)
        gt_area_error2 =np.percentile(gt_whole_area,5,axis=0)

        return area_mean,area_error1,area_error2,gt_area_mean,gt_area_error1,gt_area_error2
   
    area_mean,area_error1,area_error2,gt_area_mean,gt_area_error1,gt_area_error2 = Calculate_avg_sensitivity(Prediction[:,:,:,1:],gradient[:,:,:,1:])
    
    #Average hotspot area rate of change
    x_num = np.array([4.74,5.53,6.32,7.11,7.9,8.69,9.48,10.27,11.06,11.85,12.64,13.43,14.22,15.01])
    plt.figure (figsize= (6,4))

    plt.plot(x_num,gt_area_mean, 'b-',label='Ground truth')
    plt.plot(x_num,area_mean, 'r-',label='Prediction')

    plt.fill_between(x_num,gt_area_error1,gt_area_error2,color='blue',alpha=0.2)
    plt.fill_between(x_num,area_error1,area_error2,color='red',alpha=0.2)

    #Add labels and title
    plt.title(r'Ave. Hotspot Area Rate of Change ($\dot{A_{hs}}$)', fontsize = 14,pad = 15)
    plt.xlabel(r"t ($ns$)",fontsize = 12)
    plt.ylabel(r"$\dot{A_{hs}}$ ($\mu m^2$/$ns$)",fontsize = 12)
    plt.legend( loc = 2, fontsize = 11)
    plt.xticks(fontsize= 11)
    plt.yticks(fontsize= 11)
    plt.savefig('area_growth_plot.png')
    plt.show()

    return None


def plot_sensitivity_temperature(Prediction,gradient):
    #single sample sensitivity calculation for temperature
    def sensitivity_single_sample(test_data):
        threshold = 0.1554 #875 Temperature(K)

        area_list = []
        temp_list = []
        #Rescale the output
        test_data = (test_data+1.0)/2.0

        #Calculate area and avg hotspot
        for i in range(3,18):
            pred_slice = test_data[:,:,i]
            pred_mask = pred_slice>threshold
            pred_hotspot_area = np.count_nonzero(pred_mask)
            rescaled_area = pred_hotspot_area * ((2*25/485)**2)
            area_list.append(pred_hotspot_area)

            masked_pred = pred_slice*pred_mask
        
            if pred_hotspot_area ==0:
                pred_avg_temp=0.0
            else:
                pred_avg_temp = np.sum(masked_pred)/pred_hotspot_area
            temp_list.append(pred_avg_temp)
        return temp_list

    #average sensitivity calculation for temperature
    def Calculate_avg_sensitivity(pred,test_Y):
        whole_temp = []
        for i in range(8):
            temp_gt_list = sensitivity_single_sample(pred[i,:,:,:])
            whole_temp.append(temp_gt_list)
            
        whole_temp = np.array(whole_temp)

        temp_mean = np.mean(whole_temp,axis=0)

        temp_error1 = np.percentile(whole_temp,95,axis=0)
        temp_error2 = np.percentile(whole_temp,5,axis=0)

        gt_whole_temp = []
        
        for i in range(8):
            temp_pred_list = sensitivity_single_sample(test_Y[i,:,:,:])
            gt_whole_temp.append(temp_pred_list)
        gt_whole_temp = np.array(gt_whole_temp)

        gt_temp_mean = np.mean(gt_whole_temp,axis=0)
        
        gt_temp_error1 = np.percentile(gt_whole_temp,95,axis=0)
        gt_temp_error2 = np.percentile(gt_whole_temp,5,axis=0)
        
        return temp_mean,temp_error1,temp_error2,gt_temp_mean,gt_temp_error1,gt_temp_error2

    temp_mean,temp_error1,temp_error2,gt_temp_mean,gt_temp_error1,gt_temp_error2 = Calculate_avg_sensitivity(Prediction[:,:,:,1:],gradient[:,:,:,1:])
    
    #Average hotspot temperature rate of change
    x_num = np.array([4.74,5.53,6.32,7.11,7.9,8.69,9.48,10.27,11.06,11.85,12.64,13.43,14.22,15.01])
    plt.figure (figsize= (6,4))

    plt.plot(x_num,gt_temp_mean, 'b-',label='Ground truth')
    plt.plot(x_num,temp_mean, 'r-',label='Prediction')

    plt.fill_between(x_num,gt_temp_error1,gt_temp_error2,color='blue',alpha=0.2)
    plt.fill_between(x_num,temp_error1,temp_error2,color='red',alpha=0.2)

    #Add labels and title
    plt.title(r'Ave. Hotspot Temperature Rate of Change ($\dot{T_{hs}}$)', fontsize = 14, pad = 15)
    plt.xlabel(r"t ($ns$)", fontsize = 12)
    plt.ylabel(r"$\dot{T_{hs}}$ ($K$/$ns$)", fontsize = 12)
    plt.legend( fontsize = 11)
    plt.xticks(fontsize= 11)
    plt.yticks(fontsize= 11)
    plt.savefig('temp_growth_plot.png')
    plt.show()

    return None

def plot_saliency():
    return None
