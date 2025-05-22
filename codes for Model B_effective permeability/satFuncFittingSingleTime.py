# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:55:58 2023

@author: Xiaoming Zhang
"""
import time
from joblib import Parallel, delayed
import numpy as np
def satFuncFittingSingleTime(saturation_dataset,drainage_satTable):
    sat_num = saturation_dataset.shape[0]
    sat_xSize = saturation_dataset.shape[1]
    sat_ySize = saturation_dataset.shape[2]
    sat_zSize = saturation_dataset.shape[3]    
    
    def calculateCapi_rePerm(k):
        capi_drainage_array = np.zeros([sat_num,sat_xSize,sat_ySize])
        rePerm_drainage_array = np.zeros([sat_num,sat_xSize,sat_ySize])
        for n in range(sat_num):
            for i in range(sat_xSize):
                for j in range(sat_ySize):
                    satValue = saturation_dataset[n,i,j,k]                        
################################################################################################################
                    if drainage_satTable[n,i,j,k] == 1:
                        capi_max = 0
                        capi_min = -2000.733
                        if satValue >= 0 and satValue <= 0.6385:
                            capi_drainage = 525.337*np.power(satValue,5) - 873.65*np.power(satValue,4) + \
                            560.847*np.power(satValue,3) - 170.336*np.power(satValue,2) + 26.6902*satValue  + 0.00101204
                            capi_drainage = -capi_drainage
                            capi_drainage = min(capi_max, capi_drainage)
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                        elif satValue > 0.6385:
                            capi_drainage = 7.e22*np.power(satValue,114.065)
                            capi_drainage = -capi_drainage
                            capi_drainage = max(capi_min, capi_drainage)
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                    elif drainage_satTable[n,i,j,k] == 2: 
                        capi_max = 0
                        capi_min = -2886.67
                        if satValue >= 0 and satValue <= 0.3979:
                            capi_drainage = 3735.25*np.power(satValue,5) - 2718.7*np.power(satValue,4) + \
                            760.32*np.power(satValue,3) - 77.5917*np.power(satValue,2) + 5.43393*satValue + 0.656632
                            capi_drainage = -capi_drainage
                            capi_drainage = min(capi_max, capi_drainage)
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                        elif satValue > 0.3979 and satValue <= 0.5116:        
                            capi_drainage = 276739*np.power(satValue,3) - \
                            359192*np.power(satValue,2) + 155278*satValue - 22343.2
                            capi_drainage = -capi_drainage
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                        elif satValue > 0.5116:
                            capi_drainage = 3.e18*np.power(satValue,56.115)
                            capi_drainage = -capi_drainage
                            capi_drainage = max(capi_min, capi_drainage)
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                    elif drainage_satTable[n,i,j,k] == 3: 
                        capi_max = 0
                        capi_min = -11614.1 
                        if satValue >= 0 and satValue <= 0.1368:
                            capi_drainage = 5.52589e6*np.power(satValue,5) - 1.65412e6*np.power(satValue,4) + \
                            209368*np.power(satValue,3) - 11591.9*np.power(satValue,2) + 536.487*satValue + 0.0650038
                            capi_drainage = -capi_drainage
                            capi_drainage = min(capi_max, capi_drainage)
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                        elif satValue > 0.1368 and satValue <= 0.1895:
                            capi_drainage = 1.80795e10*np.power(satValue,5) - 1.41143e10*np.power(satValue,4) + \
                                4.40297e9*np.power(satValue,3) - 6.85882e8*np.power(satValue,2) \
                                + 5.33467e7*satValue - 1.6571e6  
                            capi_drainage = -capi_drainage
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
                        elif satValue > 0.1895:
                            capi_drainage = 1e35*np.power(satValue,44.258)
                            capi_drainage = -capi_drainage
                            capi_drainage = max(capi_min, capi_drainage)
                            capi_drainage_array[n,i,j] = (capi_drainage - capi_min)/(capi_max - capi_min)
    ################################################################################################################
                    rePerm_max = 0.86
                    rePerm_min = 0.
                    if drainage_satTable[n,i,j,k] == 1:
                        if satValue > 0:
                            rePerm_drainage = 1.71497*np.power(satValue,2) + 0.180743*satValue
                            rePerm_drainage = max(0,rePerm_drainage)
                            rePerm_drainage = min(0.86,rePerm_drainage)                               
                            rePerm_drainage_array[n,i,j] = (rePerm_drainage - rePerm_min)/(rePerm_max - rePerm_min)
                    elif drainage_satTable[n,i,j,k] == 2:
                        if satValue > 0:
                            rePerm_drainage = -0.690264*np.power(satValue,3) + \
                            2.25956*np.power(satValue,2) - 0.175653*satValue
                            rePerm_drainage = max(0,rePerm_drainage)
                            rePerm_drainage = min(0.46,rePerm_drainage)
                            rePerm_drainage_array[n,i,j] = (rePerm_drainage - rePerm_min)/(rePerm_max - rePerm_min)
                    elif drainage_satTable[n,i,j,k] == 3:
                        if satValue > 0:
                            rePerm_drainage = 136*np.power(satValue,4) - 73.929*np.power(satValue,3) \
                            + 9.4767*np.power(satValue,2) + 1.4982*np.power(satValue,1)
                            rePerm_drainage = max(0,rePerm_drainage)
                            rePerm_drainage = min(0.3,rePerm_drainage)
                            rePerm_drainage_array[n,i,j] = (rePerm_drainage - rePerm_min)/(rePerm_max - rePerm_min)
    
        return rePerm_drainage_array, capi_drainage_array
    
    t_clock = time.time()
    
    results = Parallel(n_jobs=-1)(delayed(calculateCapi_rePerm)(k) for k in range(sat_zSize))  
    results_temp = np.rollaxis(np.array(results), 0, 5)
    rePerm_drainage = results_temp[0]
    capi_drainage = results_temp[1]
    
    print(f'{time.time() - t_clock} s')
    
    return rePerm_drainage, capi_drainage
