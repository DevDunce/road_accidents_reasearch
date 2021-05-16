
"""
Created on Fri Nov 20 12:18:41 2020

@author: alanm
"""

project_path = "C:/Users/alanm/OneDrive - National College of Ireland/Final Year/Software Project/Project"

# libraries
import pandas as pd
import os
import datetime
os. chdir(project_path)


# import data
acc = pd.read_csv("./data/all_accidents.csv")




### ACCIDENT DATA ###
acc_format = acc.copy()

selected_cols = ["accident_index","longitude", "latitude", "police_force", "accident_severity", "number_of_vehicles", 
                 "number_of_casualties", "date", "day_of_week", "time", "local_authority_district", "local_authority_highway",
                 "road_type", "speed_limit", "light_conditions", "weather_conditions", "road_surface_conditions",
                 "special_conditions_at_site", "urban_or_rural_area", "did_police_officer_attend_scene_of_accident"]

# format columns dtypes

acc_format.accident_index = acc_format.accident_index.astype(str)
acc_format.date = pd.to_datetime(acc_format.date)
acc_format.info()
acc_format.accident_index.value_counts()

acc_format_cols = acc_format[selected_cols]

# find missing values
acc_format_cols.isnull().sum()
acc_format_cols.isna().sum()
acc_format_cols.isnan().sum()
acc_format_cols["accident_index"].loc[acc_format_cols["accident_index"] == "nan"].count()

acc_format_cols["accident_index"].loc[acc_format_cols["accident_index"] == "inf"].count()

acc_no_nas = acc_format_cols.loc[acc_format_cols.accident_index != "nan"]
acc_no_nas = acc_no_nas.loc[acc_format_cols.accident_index != "inf"]
acc_no_nas.accident_index.value_counts()

# trim the accident index
acc_no_nas.accident_index = acc_no_nas.accident_index.str.strip()

acc_no_nas = acc_no_nas.dropna()
acc_no_nas.isna().sum()



# remove rows with UNKNOWN values - negative value in any column (other than long-lat) - negative value indicates unknown
acc_no_negs = acc_no_nas.loc[ ~(acc_no_nas.iloc[:,2:].isin([-1]).any(axis=1))]
acc_no_negs.info()
acc_no_negs.accident_index.value_counts()
acc_no_negs.isna().sum()



# add adjustments from adjustment dataset

acc_adj = pd.read_csv("./data/all_acc_adjustments.csv")
acc_adj_severity = acc_adj.iloc[:,[1,3]]
acc_adj_severity.accident_index = acc_adj_severity.accident_index.str.strip()
acc_adj_severity.info()
acc_adj.isna().sum()


acc_adj_left_merge = pd.merge(acc_no_negs, acc_adj_severity, how="left", on="accident_index")
acc_adj_inner_merge = pd.merge(acc_no_negs, acc_adj_severity, how="inner", on="accident_index")


acc_adj_left_merge.to_csv("./data/acc_adj_left_merge.csv")
acc_adj_inner_merge.to_csv("./data/acc_adj_inner_merge.csv")



#
acc_formatter = pd.read_csv("./data/acc_adj_inner_merge.csv")
acc_formatter = pd.read_csv("./data/acc_adj_left_merge.csv")

print(acc_final.info)

str_cols = ['accident_index',  'local_authority_district',
       'local_authority_highway']

int_cols = ['police_force', 'accident_severity', 'number_of_vehicles', 'number_of_casualties','road_type', 'speed_limit',
       'light_conditions', 'weather_conditions', 'road_surface_conditions','special_conditions_at_site', 'urban_or_rural_area',
       'did_police_officer_attend_scene_of_accident', 'day_of_week']

float_cols= [ 'longitude', 'latitude']

date_cols = ["date"]
time_cols = ["time"]

dec_cols = ['adjusted_slight']

     

acc_formatter[str_cols] = acc_formatter[str_cols].astype(str)

acc_formatter[int_cols] = acc_formatter[int_cols].astype(int)

acc_formatter[float_cols] = acc_formatter[float_cols].astype(float)
acc_formatter["date"] = pd.to_datetime(acc_formatter["date"])
acc_formatter["date"] = acc_formatter["date"].dt.date

acc_formatter["time"] = pd.to_datetime(acc_formatter["time"])
acc_formatter["time"] = acc_formatter["time"].dt.time

acc_formatter["adjusted_slight"] = acc_formatter["adjusted_slight"].astype(float)



acc_formatter.to_csv("./data/acc_adj_left_merge.csv")


acc_final = pd.read_csv("./data/acc_adj_left_merge.csv")

fix_nan = acc_final.adjusted_

acc_final = acc_final.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
acc_final.to_csv("./data/final_acc.csv", index=False)


### CREATING CASUALTY DATASET ###

acc = pd.read_csv("./data/all_vehicles.csv")
veh = pd.read_csv("./data/all_vehicles.csv")
cas = pd.read_csv("./data/all_casualties.csv")

veh_head = veh.head(50)

cas_columns = ["accident_index", "vehicle_reference", "casualty_reference", "casualty_class", "sex_of_casualty",
               "age_of_casualty", "age_band_of_casualty", "casualty_severity", "pedestrian_location",
               "pedestrian_movement","car_passenger","bus_or_coach_passenger","pedestrian_road_maintenance_worker",
               "casualty_type","casualty_home_area_type"]

acc_columns = ["accident_index","number_of_vehicles","date","day_of_week","time","road_type","speed_limit","light_conditions",
               "weather_conditions","road_surface_conditions","urban_or_rural_area",
               "did_police_officer_attend_scene_of_accident"]


veh_columns = ["accident_index","vehicle_reference","vehicle_type","propulsion_code", "age_band_of_driver"]


# Keep selected columns only
cas_selected = cas[cas_columns]
acc_selected = acc[acc_columns]
veh_selected = veh[veh_columns]


# Find missing values
cas_selected.info()
cas_selected.isna().sum(axis=0)

cas_formatted = cas_selected.copy()
cas_formatted.accident_index = cas_formatted.accident_index.astype(str)

cas_formatted.accident_index.value_counts()

cas_formatted["accident_index"].loc[cas_formatted["accident_index"] == "nan"].count()
cas_formatted["accident_index"].loc[cas_formatted["accident_index"] == "inf"].count()

cas_no_nas = cas_formatted.loc[cas_formatted["accident_index"] != "nan"]
cas_no_nas = cas_no_nas.loc[cas_no_nas["accident_index"] != "inf"]

cas_no_nas.accident_index = cas_no_nas.accident_index.str.strip()

cas_no_nas = cas_selected.dropna()
cas_no_nas.isna().sum(axis=0)
cas_no_nas.accident_index.value_counts()





# Check for negative values (UNKNOWNS)

cas_negs = cas_no_nas.loc[cas_no_nas.isin([-1]).any(axis=1)]
cas_no_negs =  cas_no_nas.loc[ ~cas_no_nas.isin([-1]).any(axis=1)]

# check no (unwanted) drops
len(cas_no_nas) == len(cas_negs.index) + len(cas_no_negs)
cas_no_negs.isna().sum()


## UP TO THIS SECTION WORKS TOO



my_acc = pd.read_csv("./data/acc_adj_left_merge.csv")
my_acc_selected = my_acc[acc_columns]
my_acc_formatted = my_acc_selected.copy()


# Format my_acc columns

my_acc_formatted.accident_index = my_acc_formatted.accident_index.astype(str)
my_acc_formatted.date = pd.to_datetime(my_acc_formatted.date)

my_acc_formatted.accident_index = my_acc_formatted.accident_index.str.strip()
print(my_acc_formatted.info())

my_acc_formatted.accident_index.value_counts()



# Inner Join with accidents on accident index (having dropped casualties with no accident index)
cas_acc_left_join = pd.merge(cas_no_negs, my_acc_formatted, how="left", on="accident_index")
cas_acc_inner_join = pd.merge(cas_no_negs, my_acc_formatted, how="inner", on="accident_index")

cas_acc_left_join.to_csv("./data/cas_acc_left_join.csv")
cas_acc_inner_join.to_csv("./data/cas_acc_inner_join.csv")

## UP TO THIS SECTION WORKS TOO


### VEHICLE ###

my_cas = cas_acc_inner_join.copy()

my_cas["acc_veh_id"] = my_cas["accident_index"] + "_" + my_cas["vehicle_reference"].astype(str)

my_cas.to_csv("./data/my_cas_acc.csv")


### NEXT I NEED TO CREATE A acc_veh_id in MY_VEH, then MERGE THE VEHICLE DATA WITH THE MY_CAS (cas and acc merged) dataset


my_veh = veh_selected.copy()


# Check for missing values
my_veh.info()
my_veh.isna().sum(axis=0)

my_veh_formatted = my_veh.copy()
my_veh_formatted.accident_index = my_veh_formatted.accident_index.astype(str)
my_veh_formatted.accident_index.value_counts()


# remove NA and INF index
my_veh_formatted["accident_index"].loc[my_veh_formatted["accident_index"] == "nan"].count()
my_veh_formatted["accident_index"].loc[my_veh_formatted["accident_index"] == "inf"].count()

my_veh_formatted = my_veh_formatted.loc[my_veh_formatted["accident_index"] != "nan"]
my_veh_formatted = my_veh_formatted.loc[my_veh_formatted["accident_index"] != "inf"]

my_veh_formatted["accident_index"] = my_veh_formatted["accident_index"].str.strip()





my_veh_with_id = my_veh_formatted.copy()

my_veh_with_id["acc_veh_id"] = my_veh_with_id["accident_index"] + "_" + my_veh_with_id["vehicle_reference"].astype(str)

my_veh_with_id.to_csv("./data/my_veh_with_id.csv")

cas_acc_veh_left = pd.merge(my_cas, my_veh_with_id, how="left", on="acc_veh_id")
cas_acc_veh_inner = pd.merge(my_cas, my_veh_with_id, how="inner", on="acc_veh_id")

cas_acc_veh_left.to_csv("./data/cas_acc_veh_left.csv")
cas_acc_veh_inner.to_csv("./data/cas_acc_veh_inner.csv")


## SO FAR SO GOOD

# Bring in casualty severity adjustments finally

cas_adj = pd.read_csv("./data/adjustments/cas_adjustment_lookup_2019.csv")


cas_adj_format = cas_adj.copy()

cas_adj_format.accident_index = cas_adj_format.accident_index.astype(str)

cas_adj_format.accident_index.value_counts()

cas_adj_format["acc_veh_cas_ref"] = cas_adj_format["accident_index"] + "_" + cas_adj_format["Vehicle_Reference"].astype(str) + "_" + cas_adj_format["Casualty_Reference"].astype(str)

cas_adj_format = cas_adj_format.rename(str.lower, axis="columns")

cas_adj_cols = ["acc_veh_cas_ref","adjusted_slight"]

final_cas_adj = cas_adj_format[cas_adj_cols]
final_cas_adj.to_csv("./data/final_cas_adj.csv")

casualty_data_with_full_reference = cas_acc_veh_inner.copy()


## JUSTR NEED TO FINISH HERE



final_cas = pd.read_csv("./backup/cas_acc_veh_inner.csv")

final_cas = final_cas.drop(["accident_index_y", "vehicle_reference_y"], axis=1)

final_cas = final_cas.rename(columns={"accident_index_x": "accident_index", "vehicle_reference_x" : "vehicle_reference"}, errors="raise")

final_cas = final_cas.rename(columns={"vehicle_reference_x" : "vehicle_reference"}, errors="raise")


final_cas["acc_veh_cas_ref"] = final_cas["accident_index"].astype(str) + "_" + final_cas["vehicle_reference"].astype(str) + "_" + final_cas["casualty_reference"].astype(str)

final_cas_before_merge = final_cas.copy()
final_cas_before_merge.to_csv("./data/final_cas_before_merge.csv")




final_complete_cas = pd.merge(final_cas, final_cas_adj, how="inner", on="acc_veh_cas_ref")


final_complete_cas.to_csv("./data/final_cas_dataset.csv")
final_complete_cas.to_csv("./backup/final_cas_dataset.csv")



project_path = "C:/Users/alanm/OneDrive - National College of Ireland/Final Year/Software Project/Project"



# libraries
import pandas as pd
import numpy as np
import os
import datetime
import math
os. chdir(project_path)


final_acc = pd.read_csv("./backup/final_acc.csv")
final_cas = pd.read_csv("./backup/final_cas.csv", low_memory=False)

for c in final_cas.columns:
    print(c, final_cas[c].nunique())
    
for c in final_acc.columns:
    print(c, final_acc[c].nunique())
    

final_acc.accident_index = final_acc.accident_index.astype(str)
    
acc_head = final_acc.sample(n=100, random_state=1)
cas_head = final_cas.sample(n=100, random_state=1)

acc_head.to_excel("./data/acc_head.xlsx", index=False)
cas_head.to_excel("./data/cas_head.xlsx", index=False)
