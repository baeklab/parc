import os
import numpy as np
from PIL import Image
import cv2
import skimage

def parseData(file_name: str) -> np.ndarray:
    """parse the raw data and return numpy arary

    Args:
        file_name (str):    file name of raw data

    Returns:
        processedData (np.ndarray): processed data
    """
    input_data = []
    output_data = []

    #Generate Distance map (normalized distance from y-axis)
    #Size of the map is same as the original microsturcture image size (485x485)
    wave_map = np.zeros((485,485))
    for w in range(1,485):
        wave_map[:,w] = w/485.0

    #There are currenly 36 samples that can be used
    for i in range(1,37):

        input_images = []

        #Load Original Microstructure Image
        original_img = './' + file_name + '/microstructures/data_'+str(format(i,'02d'))+'.pgm'
        x = cv2.imread(original_img)
        ori_img = x[:,:,1]

        #Combine Microstructure image with distance map
        input_images.append(ori_img)
        input_images.append(wave_map)
        input_images = np.array(input_images)
        input_images = np.rollaxis(input_images, 0, 3)
        input_data.append(input_images)

        #Load,reshape and clip the temperature data
        out_whole = []
        for k in range(0,20):
            if k ==0:
                y = np.full((485,485),300.)
                z = np.full((485,485),0)
            else:
                temperature_name =  './data/raw/temperatures/data_'+str(format(i,'02d'))+'/Temp_'+str(format(k,'02d'))+'.txt'
                temperature_img = np.loadtxt(temperature_name)
                y = np.reshape(temperature_img,(485,485))

                pressure_name =  './data/raw/pressures/data_'+str(format(i,'02d'))+'/pres_'+str(format(k,'02d'))+'.txt'
                pressure_img = np.loadtxt(pressure_name)
                z = np.reshape(pressure_img,(485,485))

                #clip the temperature value such that it ranges between 300K and 4000K
                y = np.clip(y,300,4000)

            out_whole.append(y)
            out_whole.append(z)
              
        out_whole = np.array(out_whole)
        out_whole = np.rollaxis(out_whole, 0, 3)
        output_data.append(out_whole)


    #Make input data and output data to array
    data_in = np.array(input_data)
    output_data = np.array(output_data)


    #Calculate T_dot --> T_dot = (T(t+del_t)-T(t))/del_t
    #del_t we use is approximately 0.79 e-9

    whole_dot = []
    del_t = 0.79*(10**-9)
    for y in range(0,38,2):

        Tdot = output_data[:,:,:,y+2] - output_data[:,:,:,y]
        Pdot = output_data[:,:,:,y+3] - output_data[:,:,:,y+1]

        whole_dot.append(Tdot)
        whole_dot.append(Pdot)

    whole_dot = np.array(whole_dot)
    concat_dot= np.rollaxis(whole_dot, 0, 4)
    concat_dot = concat_dot/del_t


    #Normalize Temperature to range [-1,1]
    norm_T_max = np.amax(output_data[:,:,:,0::2])
    norm_T_min = np.amin(output_data[:,:,:,0::2])
    output_data[:,:,:,0::2] = ((output_data[:,:,:,0::2]-norm_T_min)/(norm_T_max-norm_T_min))
    output_data[:,:,:,0::2] = (output_data[:,:,:,0::2] * 2.0)-1.0
    print('max and min of original temperature data are: ',norm_T_max,norm_T_min)


    #Normalize Pressure to range [-1,1]
    norm_P_max = np.amax(output_data[:,:,:,1::2])
    norm_P_min = np.amin(output_data)
    output_data[:,:,:,1::2] = ((output_data[:,:,:,1::2]-norm_P_min)/(norm_P_max-norm_P_min))
    output_data[:,:,:,1::2] = (output_data[:,:,:,1::2] * 2.0)-1.0
    print('max and min of original Pressure data are: ',norm_P_max,norm_P_min)

    #Normalize T_dot to range [-1,1]
    norm_Tdot_max = np.amax(concat_dot[:,:,:,0::2])
    norm_Tdot_min = np.amin(concat_dot[:,:,:,0::2])
    concat_dot[:,:,:,0::2] = ((concat_dot[:,:,:,0::2]-norm_Tdot_min)/(norm_Tdot_max-norm_Tdot_min))
    concat_dot[:,:,:,0::2] = (concat_dot[:,:,:,0::2]*2.0)-1.0
    print('max and min of original Tdot data are: ',norm_Tdot_max,norm_Tdot_min)

    #Normalize P_dot to range [-1,1]
    norm_Pdot_max = np.amax(concat_dot[:,:,:,1::2])
    norm_Pdot_min = np.amin(concat_dot[:,:,:,1::2])
    concat_dot[:,:,:,1::2] = ((concat_dot[:,:,:,1::2]-norm_Pdot_min)/(norm_Pdot_max-norm_Pdot_min))
    concat_dot[:,:,:,1::2] = (concat_dot[:,:,:,1::2]*2.0)-1.0
    print('max and min of original Pdot data are: ',norm_Pdot_max,norm_Pdot_min)

    #Normalize input data to range [-1,1]
    data_in[:,:,:,0] = (data_in[:,:,:,0]-np.amin(data_in[:,:,:,0]))/(np.amax(data_in[:,:,:,0])-np.amin(data_in[:,:,:,0]))
    data_in[:,:,:,1] = (data_in[:,:,:,1]-np.amin(data_in[:,:,:,1]))/(np.amax(data_in[:,:,:,1])-np.amin(data_in[:,:,:,1]))
    data_in[:,:,:,0] = data_in[:,:,:,0] >0.5
    data_in = (data_in*2.0)-1.0

    #concatenate Temperature and T_dot
    total_output = np.concatenate((output_data,concat_dot),axis=-1)

    print('Finished Processing Data')
    print('shape of input data is: ',data_in.shape)
    print('shape of output data is: ',total_output.shape)

    np.save("output_data",total_output)
    np.save("input_data",data_in)
    return None

def getData(data_dir: str, splits: list) -> np.ndarray:
    """parse the data and return data

    Args:
        file_name (str):    file name of raw data
        splits (list[int]): train, val, test split

    Returns:
        processedData (np.ndarray): processed data
    """
    #Load preprocessed temperature and pressure values
    total_output = np.load("output_data.npy")
    data_in = np.load("input_data.npy")
    total_output = total_output[:,:480,:480,:]
    data_in = data_in[:,:480,:480,:]
    
    # downsample to 240x240
    total_output = skimage.measure.block_reduce(total_output, (1,2,2,1),np.max)
    data_in = skimage.measure.block_reduce(data_in, (1,2,2,1),np.mean)
    data_in[:,:,:,:1] = data_in[:,:,:,:1]>0
    data_in[:,:,:,:1] = (data_in[:,:,:,:1]*2.0)-1.0
    
    # Training
    X_train = np.concatenate((data_in[:10,:,:,:],data_in[14:24,:,:,:],data_in[28:38,:,:,:]),axis = 0)
    y_train = np.concatenate((total_output[:10,:,:,2:],total_output[14:24,:,:,2:],total_output[28:38,:,:,2:]),axis = 0)
    X_train_init = np.concatenate((total_output[:10,:,:,0:2],total_output[14:24,:,:,0:2],total_output[28:38,:,:,0:2]),axis = 0)

    #Validation
    X_val = np.concatenate((data_in[10:11,:,:,:],data_in[24:25,:,:,:],data_in[38:39,:,:,:]),axis = 0)
    y_val = np.concatenate((total_output[10:11,:,:,2:],total_output[24:25,:,:,2:],total_output[38:39,:,:,2:]),axis = 0)
    X_val_init = np.concatenate((total_output[10:11,:,:,0:2],total_output[24:25,:,:,0:2],total_output[38:39,:,:,0:2]),axis = 0)

    # Test
    test_X = np.concatenate((data_in[11:14,:,:,:],data_in[25:28,:,:,:],data_in[39:,:,:,:]),axis = 0)
    test_Y = np.concatenate((total_output[11:14,:,:,2:],total_output[25:28,:,:,2:],total_output[39:,:,:,2:]),axis = 0)
    test_X_init = np.concatenate((total_output[11:14,:,:,0:2],total_output[25:28,:,:,0:2],total_output[39:,:,:,0:2]),axis = 0)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(test_X.shape)
    print(test_Y.shape)
    return None, None, None, None, None, None
