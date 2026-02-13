#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:24:35 2025
Data steady-state-er
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import walk
from Time_stamper_v1 import time_stamper
#from Data_writer_v1 import data_writer
import numpy as np
import csv
import os
import sys

import json

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    """Resolve the *reference* folder robustly.

    Walk up the path to find known modality markers ("PLI", "PI", "SAXS", "Rheology").
    The *reference* folder is the parent directory of the modality folder.
    Fallback: two levels up from the provided path (legacy behavior).
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    abspath = os.path.abspath(path)
    parts = _split_parts(abspath)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            reference = os.sep.join(parts[:i])
            if reference == "":
                break
            return reference

    return os.path.dirname(os.path.dirname(abspath))


def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    os.makedirs(processed_root, exist_ok=True)
    return processed_root


Parent_folder = '/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/_Papers/2025_Rheo_PI_SAXS/_DATA/PS_15min'

Viscosity = '/Viscosity'

SAXS = '/SAXS'

PI = '/PI'

SS = '/Results'
    
#print(Trigger)

Root = Parent_folder + SAXS

Triggers = np.zeros((29, 3))

res = []
counter = 0
for (source, dir_names, file_names_trig) in walk(Root):
    for file_name_trig in file_names_trig:
        if file_name_trig.endswith(".txt") and file_name_trig.startswith("time"):
            res = os.path.join(source, file_name_trig)
            print(file_name_trig)
            Triggers[:, counter] = time_stamper(res)
            counter += 1
   
Root = Parent_folder + '/' + '_Results_triggers/'  
   
df_means = pd.DataFrame(Triggers)
Name_with_path= Root + 'triggers.csv'
df_means.to_csv(Name_with_path , index=False, header=False)

processed_folder = get_unified_processed_folder(Parent_folder)
print(f"Unified _Processed folder for Steady Stater outputs: {processed_folder}")

json_path = os.path.join(processed_folder, "_output_Steady_Triggers.json")
with open(json_path, "w") as f:
    json.dump({
        "parent_folder": Parent_folder,
        "processed_folder": processed_folder,
        "trigger_csv": Name_with_path
    }, f, indent=2)
print(f"Steady Stater trigger JSON written: {json_path}")

x = np.arange(0,np.shape(Triggers)[0])

plt.figure(figsize=(10, 6))
plt.scatter(x,Triggers[:,0], label='1')
plt.scatter(x,Triggers[:,1], label='2')
plt.scatter(x,Triggers[:,2], label='3')
plt.xlabel('Index')
plt.ylabel('Times at steady state')
plt.title('Second and Third Columns as a function of Time')
plt.legend()
plt.show()

File_names=[]

Root = Parent_folder + '/' + '_Results_PI_transient'     

for file in sorted(os.listdir(Root)):
     if file.endswith("tard_avg_time.csv") or file.endswith("ent_avg_time.csv"):
         print(file)
         File_names.append(file)
         
#print(File_names)
         
size=np.shape(File_names)[0] 

for j in range(0, size):
    path_to_filename=Root+'/'+ File_names[j] 
    Data = pd.read_csv(path_to_filename)
    if j in range(1,2):
        Temp_Trig = Triggers[:,1]
    else: Temp_Trig = Triggers[:,0]

    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data)

    No_of_col = Temp_Trig.shape[0]
    avg_values = np.zeros((No_of_col, Data.shape[1]-1))
    
    Data.iloc[:, 0] = Data.iloc[:, 0] - Data.iloc[0, 0]
    
    Data_temp = Data.to_numpy()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(Data_temp[:,0],Data_temp[:,1], label='2')
    plt.xlabel('Index')
    plt.ylabel('Times at steady state')
    plt.title('Second plot')
    plt.legend()
    plt.show()
    
    counter=0

    #print(Data)
    #Data_temp[:,0]=Data_temp[:,0]-Data_temp[0,0]
        
    for i in range(1, No_of_col):
            start_time = Temp_Trig[i]
            end_time = start_time + 4
            mask = (Data.iloc[:, 0] >= start_time) & (Data.iloc[:, 0] < end_time)
            avg_values[i, 0] = Data.loc[mask, Data.columns[1]].mean()
            avg_values[i, 1] = Data.loc[mask, Data.columns[2]].mean()
        
    #avg_values
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_values[:,0], label='Second Column')
    plt.plot(avg_values[:,1], label='Third Column')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(File_names[j])
    plt.legend()
    plt.show()
    
    
    Sub_Folder = "/_Results_PI_steady_state/"
    
    Results_paths = Parent_folder + Sub_Folder
    
    Names_to_write = File_names[j][:-4]
    df_means = pd.DataFrame(avg_values)
    #Temp1_avg=Means2.to_numpy()
    Name_with_path= Results_paths + Names_to_write + '_steady_state.csv'
    df_means.to_csv(Name_with_path , index=False, header=False)
    
    json_path = os.path.join(processed_folder, f"_output_Steady_{os.path.splitext(File_names[j])[0]}.json")
    with open(json_path, "w") as f:
        json.dump({
            "source_file": File_names[j],
            "processed_folder": processed_folder,
            "steady_state_output": Name_with_path
        }, f, indent=2)
    print(f"Steady Stater output JSON written: {json_path}")
    
    #return 

# data_path = '/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/_Papers/2025_Rheo_PI_SAXS/_DATA/PS_15min/PI/CSVs/_Results/20250222_112802_CNC4 PS 15MINS SS_avg_time.csv'

# trigger_path = '/Users/kroland/Library/Mobile Documents/com~apple~CloudDocs/_Papers/2025_Rheo_PI_SAXS/_DATA/PS_15min/SAXS/CNC4_PS15_0.3(tan)/time_46814.txt'

# Trigger = Time_stamper_v1(trigger_path)

# Data = pd.read_csv(data_path)

# test = steady_stater_v2(Data,Trigger)

# plt.figure(figsize=(10, 6))
# for i in range(1, test.shape[1]):
#     plt.scatter(range(test.shape[0]), test[:, i], label=f'Column {i+1}', facecolors='none', edgecolors='b')
# plt.xlabel('Index')
# plt.ylabel('Average Values')
# plt.title('Steady State Points')
# plt.legend()
# plt.show()
