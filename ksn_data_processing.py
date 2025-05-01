# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:36:04 2023

@author: mjrobinson
"""



import numpy as np
import pandas as pd


# This code will extract ksn values from lsdtopotools output


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
    # read in lsdtopotools mchidata
    
    df1 = pd.read_csv('{}lsd_MChiSegmented.csv'.format(
        filelist[h]), header=0, float_precision='round_trip')

    df1 = df1.rename(columns={'basin_key': "basin_id"})
    df1 = df1.rename(columns={'m_chi': "ksn"})
    channel_data = df1
    del channel_data['node']
    del channel_data['row']
    del channel_data['col']
    # del channel_data['flow_distance']
    del channel_data['elevation']
    del channel_data['chi']
    del channel_data['latitude']
    del channel_data['longitude']
    # del channel_data['drainage_area']
    del channel_data['b_chi']
    del channel_data['source_key']
    del channel_data['segmented_elevation']

    channel_data.columns = ['chi', 'drainage_area', 'z',
                            'flowdistance', 'source', 'StreamOrder', 'basin_id', 'x', 'y']

    # %%
# make list of seperate basins
    channel_data_round = channel_data.astype('int')
    rng = channel_data_round.loc[channel_data_round.groupby(
        'basin_id')['drainage_area'].nlargest(1).reset_index(0).index]

    basin_list = np.array(rng['basin_id'])

    print("this is {} basin list".format(namelist[h]))

 
    # %%
    # counters that are usefull for saving data. 

    # cccount increases by 1 each time through the loop

    cccount = 0
 
    # arrays to put ksn data into. 
    saveksn_mean = np.zeros(np.size(basin_list))
    saveksn_std = np.zeros(np.size(basin_list))

 # %%
# loop through all basins in the landscape
    for f in range(0, np.size(basin_list), 1):
        

        basinpick = basin_list[f]
        
        # river data for basin 

        for key, grp in channel_data.groupby(['basin_id']):
            if key == basinpick:
                savearea= grp["drainage_area"].max()
               
                flowchannel = np.array(grp['flow_distance'])
                saveksn_mean[cccount] = grp["ksn"].median()
                saveksn_std[cccount] = grp["ksn"].std()

        # check to ensure basin size meets expectation. LSDTopotools has a bug that tacks on small basins significantly below specified threshold size after the outlet of other basins


        if savearea<(1*10**6):
            print("this is to small")
            print(basinpick)
            
            continue 
 
        if np.size(flowchannel) < 4:
            continue

    # remove zeros caused by basins being skipped
    saveksn_std = saveksn_std[saveksn_mean != 0]
    saveksn_mean = saveksn_mean[saveksn_mean != 0]
 




      