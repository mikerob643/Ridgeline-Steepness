# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:36:04 2023

@author: mjrobinson
"""


from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from kneed import KneeLocator
from statsmodels.stats.stattools import durbin_watson
from scipy.optimize import curve_fit

# This code will extract ridgelines from a landscape and  fit a power law to the
# ridgeline slope, basin area data. The code uses basin and river csv files created using the lsdtopotools data and qgis processing 
# described in previous steps.


# location of basin and river data files, names from paper are left as placeholders
filelist = ["WestVirgina/", 
           
            "OCR/"
            ]
# names you want associated with landscapes

namelist = ["Allegheny Plateau, WV",
         
            "Oregon Coast Range, OR"
           ]

# abreviated names


ab = ["wv", 
    
      "ocr"
    ]


def calculate_cumulative_distance(points):
    """Calculates the cumulative distance between consecutive points."""
    if points.shape[0] < 2:
        return np.array([0.0] * (points.shape[0] + 1)) if points.shape[0] >= 0 else np.array([0.0])
    diffs = np.diff(points[:, 0:2], axis=0)
    squared_distances = np.sum(diffs**2, axis=1)
    distances_between_points = np.sqrt(squared_distances)
    cumulative_distances = np.cumsum(distances_between_points)
    return np.concatenate(([0.0], cumulative_distances))




# loop through landscapes 
for h in range(np.size(filelist)):
    
    

# read in basin data


    df = pd.read_csv('{}basins.csv'.format(
        filelist[h]), header=0, float_precision='round_trip')

    df = df.rename(columns={'fid': "basin_id"})
    basin_data = df
    
    
#delete unwanted columns
    
    
    del basin_data['DN']

    del basin_data['distance']
    del basin_data['angle']
# rename columns
    basin_data.columns = ['basin_id', 'x', 'y', 'z']
    basin_data = basin_data.dropna()
    
    
# read in channel data
    df1 = pd.read_csv('{}channel1.csv'.format(
        filelist[h]), header=0, float_precision='round_trip')

    df1 = df1.rename(columns={'fid': "basin_id"})
    df1 = df1.rename(columns={'elevation': "z"})
    channel_data = df1
    del channel_data['feature_x']
    del channel_data['feature_y']
    del channel_data['nearest_x']
    del channel_data['nearest_y']
    del channel_data['distance']
    del channel_data['basin_key']
    del channel_data['latitude']
    del channel_data['longitude']
    
    
# del channel_data['StreamOrder']
    del channel_data['n']
    del channel_data['DN']

    channel_data.columns = ['chi', 'drainage_area', 'z',
                            'flowdistance', 'source', 'StreamOrder', 'basin_id', 'x', 'y']

    # %%
# make list of seperate basins

    basin_list = channel_data['basin_id'].unique()
    
    print("this is {} basin list".format(namelist[h]))

    # %%

# create a data storage plan here. below are arrays for basic info only
  
    savearea = np.zeros((np.size(basin_list))*2)
    slopeofridge = np.zeros((np.size(basin_list))*2)

 
    # %%
    # counters that are usefull for saving data. 
    # cccount increases by 1 each time through the loop
    # matrixcount increases by 2, usefull for ridgeline data which has two data points per basin usually. 
    cccount = 0
    matrixcount = 0



 # %%
# loop through all basins in the landscape
    for f in range(0, np.size(basin_list), 1):
        

        basinpick = basin_list[f]

        
        # river data for basin 
        for key, grp in channel_data.groupby(['basin_id']):
            if key == basinpick:
                # there are two features per basin so the area counts twice 
                area = grp["drainage_area"].max()
            
                
                xchannel = np.array(grp['x'])
                ychannel = np.array(grp['y'])
                zchannel = np.array(grp['z'])
                flowchannel = np.array(grp['flowdistance'])
          
                
                chi = np.array(grp['chi'])

                sources = np.array(grp['source'])

                zchannel = (np.array(grp['z']))
                ychannel = (np.array(grp['y']))
                source = (np.array(grp['source']))

          
        # check to ensure basin size meets expectation. LSDTopotools has a bug that tacks on small basins significantly below specified threshold size after the outlet of other basins

        if area<(1*10**6):
            print("thisistosmall")
            print(basinpick)
     
            
            continue 
        # bring in ridge data for basin
        for key, grp2 in basin_data.groupby(['basin_id']):
            if key == basinpick:
         
             
                xridge = np.array(grp2['x'])
                yridge = np.array(grp2['y'])
                zridge = np.array(grp2['z'])
                
                
        # again check to ensure basin size meets expectation. LSDTopotools has a bug that tacks on small basins significantly below specified threshold size after the outlet of other basins
      
        # these checks appeared to be enough to resolve the issue
        if np.size(yridge)<10:
            print("thisbasin doesnt have enough points")
            print(basinpick)
       
            continue 
        
       
        if np.size(xchannel) < 4:
            print("does this happen?")
         
            continue
        
        # check to make sure nothing is odd with elevations. 
        if np.max(zridge)>8000:
            print("something is up with the height")
            print(basinpick)
       
            continue 




        # %%

        channeldata = np.empty(shape=(len(xchannel), 3))
        channeldata[0:, 0] = xchannel
        channeldata[0:, 1] = ychannel
        channeldata[0:, 2] = flowchannel

        ridgedata = np.empty(shape=(len(xridge), 3))
        ridgedata[0:, 0] = xridge
        ridgedata[0:, 1] = yridge

        # Now split the perimeter into two seperate ridgelines
        #start with lowest point
        lowest_point_on_ridge = np.where(zridge == np.amin(zridge))

        # ok this is the lowest ridge, split array around this.

        reoganizeridge = ridgedata[((lowest_point_on_ridge[0][0])):, 0:]
        reoganize2 = ridgedata[0:(lowest_point_on_ridge[0][0]), 0:]
        ridgedata = np.concatenate((reoganizeridge, reoganize2))

        reorganizez = zridge[((lowest_point_on_ridge[0][0])):]
        zridge = np.append(
            reorganizez, zridge[0:(lowest_point_on_ridge[0][0])])



        # %%  #find closest ridge point to the top of the the mainstem channel to make the top cut
        

        mainstemx=xchannel[source==np.min(source)]
        mainstemy=ychannel[source==np.min(source)]
        mainstemz=zchannel[source==np.min(source)]



        mainstem_x_start = mainstemx[0]
        mainstem_y_start = mainstemy[0]
        num_points = len(zridge)  
  
        ridgedata_segment = ridgedata[:num_points, 0:2]  # Assuming x is col 0, y is col 1

        squared_diff_x = (ridgedata_segment[:, 0] - mainstem_x_start)**2
        squared_diff_y = (ridgedata_segment[:, 1] - mainstem_y_start)**2
        
        # Calculate the Euclidean distances
        distance = np.sqrt(squared_diff_x + squared_diff_y)
        
      
        furthest_distance_index = np.where(distance == np.min(distance))
     
        
        if np.size(furthest_distance_index) == 1:
       
            furthest_distance_index = furthest_distance_index[0][0]
            
        # if multiple points along the ridge have the same distance to the mainstem channel head, take the average index or the middle ridgeline point
        if np.size(furthest_distance_index) > 1:
                
                for e in range(np.size(furthest_distance_index)):
                    furthest_distance_index=np.average(furthest_distance_index)
                    furthest_distance_index=int(furthest_distance_index)
        
        
        # add in an identifier to determine left or right. ridge 1 =1 two =2
        ridgedata[0:furthest_distance_index, 2] = 1
        ridgedata[furthest_distance_index:, 2] = 2
    
        # now the ridges are split into two 
        
        rel1 = zridge[(np.where(ridgedata[0:, 2] == 1))]
        rel2 = zridge[(np.where(ridgedata[0:, 2] == 2))]
        

 
        ridge_column_index = 2
        
        # Get indices for each ridge
        ridge1_indices = np.where(ridgedata[:, ridge_column_index] == 1)[0]
        ridge2_indices = np.where(ridgedata[:, ridge_column_index] == 2)[0]
   
        if ridge1_indices.size > 0:
            ridge1_points = ridgedata[ridge1_indices, :]
            distance1 = calculate_cumulative_distance(ridge1_points)
        else:
            distance1 = np.array([0.0])
        
        if ridge2_indices.size > 0:
            ridge2_points = ridgedata[ridge2_indices, :]
            distance2 = calculate_cumulative_distance(ridge2_points)
        else:
            distance2 = np.array([0.0])
        
        
    
        
        # When converting the allbasins raster to polygons then points along geometry, 
        # sometimes qgis does not produce ridgeline points in order. This bug is rare but 
        # if it happens we kick this basin out of the analysis. 
        if np.max(distance1[1:] - distance1[:-1])>100:

            continue 

        if np.max(distance2[1:] - distance2[:-1])>100:
            
            continue 
    

        #%%
        # here we find xcmax using on the smoothed data and then measure the ridgeline slope up ridge of that point. 
        noisy_data=rel1[rel1>5+np.min(rel1)] 
        in_array=distance1[rel1>5+np.min(rel1)]
   
     
        lowess_tight = lowess(noisy_data, in_array, frac = .2)

        x_raw=lowess_tight[:,0]
        y_raw=lowess_tight[:,1]
        
        if np.size(lowess_tight[:,1]>0):
        
            kn1 = KneeLocator(lowess_tight[0:,0],lowess_tight[:,1])

            threshold=y_raw[kn1.knee==x_raw]
            
            if len(threshold)==0:
                print("huh")
         
            xthreshold1=x_raw[kn1.knee==x_raw]/np.max(distance1)

        if np.size(threshold) > 0:
            this = (np.where(rel1 > threshold)[0])
            if np.size(this) > 0:
                 
       
    
                testx1 = distance1[(np.where(rel1 > threshold)[0][0]):]

                testy1 = rel1[(np.where(rel1 > threshold)[0][0]):]
    
           
                if np.size(testx1) > 15:
                    indev5 = sm.add_constant(testx1)
                    results = sm.OLS(testy1, indev5).fit()
                    durban_watson_statistic = durbin_watson(results.resid)
                    
          
                
     
   
                    savearea[matrixcount] = area
    
                    ridgeslope1 = results.params[1]
                    slopeofridge[matrixcount] = ridgeslope1
               
               
                    
                    
             
                    fit=np.polyfit(testx1,testy1,2)
           
                    curvature_ratio= np.abs(fit[1] / fit[0])  # Ratio of linear and nonlinear terms in quadratic, refered to as beta

                    
            
              
# ridge two has an inverted distance so we take care of that here 
        inverted_distance2=np.max(distance2)-distance2
        noisy_data=rel2[rel2>5+np.min(rel2)]
        in_array=inverted_distance2[rel2>5+np.min(rel2)]
   
 
   
        lowess_tight = lowess(noisy_data, in_array, frac = .2)


        x_raw=lowess_tight[:,0]
        y_raw=lowess_tight[:,1]
        
        if np.size(lowess_tight[:,1]>0):
         
            kn1 = KneeLocator(lowess_tight[0:,0],lowess_tight[:,1])



            threshold=y_raw[kn1.knee==x_raw]
          
            if len(threshold)==0:
                print("huh")
             
            xthreshold2=x_raw[kn1.knee==x_raw]/np.max(distance2)

        if np.size(threshold )>0:
            this = (np.where(rel2 > threshold)[0])
            if np.size(this) > 0:
             
    
                testx2 = inverted_distance2[(np.where(rel2 > threshold)[0][0]):]
              
                testy2 = rel2[(np.where(rel2 > threshold)[0][0]):]
    
           
                if np.size(testx2) > 15:
                    indev5 = sm.add_constant(testx2)
                    
                    results = sm.OLS(testy2, indev5).fit()
                    durban_watson_statistic = durbin_watson(results.resid)
             
                
             
                        
                    savearea[matrixcount+1] = area
                    ridgeslope2 = results.params[1]
                    slopeofridge[matrixcount+1] = ridgeslope2
       
             
                    
                    fit=np.polyfit(testx1,testy1,2)
                    curvature_ratio= np.abs(fit[1] / fit[0])  # Ratio of linear and nonlinear terms in quadratic, refered to as beta




# %%

        matrixcount = matrixcount+2
        cccount = cccount+1
        
#%%
# if basin or ridgeline was skipped this removes resulting the 0 values from array. 
    savearea=savearea[slopeofridge != 0]
    savearea_filtered = savearea

       
    slopeofridge = slopeofridge[slopeofridge != 0]
    
    # %% area slope for each landscaped 


# values derived from curve_fit of func can be used to find the best fit theta ref ridgeline 
    def func(x,a,n):
        return a*(x**(n))

    popt, pcov = curve_fit(func, savearea,slopeofridge)
    
    exponent = popt[1]
    constant= popt[0]
    perr = np.sqrt(np.diag(pcov))
    
    

    g= len(slopeofridge) # Number of data points
    w = len(popt) 
    exponent_95 = stats.t.interval(0.95, g-w, loc=popt[1], scale=perr[1])[1]


    
 
# measuring normalized ridgeline steepness with theta ref set to 0.45

    gamma=np.mean(slopeofridge/(savearea**(-0.45)))
    
    gammastd=np.std(slopeofridge/(savearea**(-0.45)))
 

 











