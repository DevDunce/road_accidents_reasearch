# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:33:44 2021

@author: alanm
"""

""" ENTER UPDATED PROJECT PATH """

project_path = "C:/Users/alanm/OneDrive - National College of Ireland/Final Year/Software Project/Project"

# libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import janitor
import numpy as np


plt.style.use("classic")
os.chdir(project_path)


# import data
full_set = pd.read_csv("./data/ALL_RECORDS.csv")

for c in full_set.columns:
    print(c, full_set[c].nunique())
    
    
full_set.describe()




# pickling
# with open ("./data/pickles/full_dataset.pkl", "wb") as f:
#     pickle.dump(full_set, f)

# with open("./data/pickles/full_dataset.pkl", "rb") as f:
#     full_set = pickle.load(f)


## Prepare Initial Full Set -------------------------

# format column headers
full_set.columns = full_set.columns.str.replace("Vehicles.", "")
full_set = full_set.clean_names()
full_set.columns



# full_set.to_csv("./data/full_dataset.csv", index=False)



# Feature Selectin Review -----------------------------------------------

# should do this at an earlier stage, but testing here on the prepared set

target_var = "casualty_severity"
dropping_var = "accident_severity"

df = full_set.copy()

df = df.select_dtypes(['number'])
                                                          
# stratified split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 1)

for train_index, test_index in split.split(df, df[target_var]):
    train = df.loc[train_index]
    test = df.loc[test_index]


## Separate Labels

# predictors (X)
X_train = train.drop(target_var, axis=1)
X_test= test.drop(target_var, axis=1)

# labels (y)
y_train = train[target_var].copy()
y_test = test[target_var].copy()








# FILTERING BY YEAR

allyears = full_set.copy()


allyears["year"] = pd.DatetimeIndex(allyears["date"]).year

year18 = allyears.copy()
year19 = allyears.copy()

year18 = year18.loc[year18["year"] == 2018]
year19 = year19.loc[year19["year"] == 2019]






## PREPARATION ---------------------------------------------------------------------------------

""" THIS IS THE DATASET THAT WILL BE RUN THROUGH THE PIPELINE"""

df_to_be_prepared = allyears
outfile_name = "allyears_no_eng"
df_to_be_prepared.drop("year", axis=1)

# df_to_be_prepared.to_csv("./data/full_data_before_prep.csv", index=False)

"""-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-"""

# Dropping columns ----------

drop_cols =[
"source_name",
"accident_index",
"vehicle_reference",
"casualty_reference",
"casualty_class",
"pedestrian_location",
"pedestrian_movement",
"car_passenger",
"bus_or_coach_passenger",
"pedestrian_road_maintenance_worker",
"location_easting_osgr",
"location_northing_osgr",
"1st_road_class",
"1st_road_number",
"junction_control",
"2nd_road_class",
"2nd_road_number",
"pedestrian_crossing_human_control",
"pedestrian_crossing_physical_facilities",
"special_conditions_at_site",
"carriageway_hazards",
"towing_and_articulation",
"vehicle_location_restricted_lane",
"hit_object_in_carriageway",
"vehicle_leaving_carriageway",
"hit_object_off_carriageway",
"was_vehicle_left_hand_drive_",
"journey_purpose_of_driver",
"propulsion_code"
]

reduced_set = df_to_be_prepared.drop(drop_cols, axis=1)

# Dropping Missing Values ----------

# drop records with any values as -1 (missing information)
reduced_set.isna().sum()


no_na = reduced_set.fillna(-1)
no_na.isna().sum()


# drop unknown sex_of_driver values with 3 as gender
no_na.sex_of_driver.value_counts()
no_na.sex_of_driver = no_na.sex_of_driver.replace(3,-1)
no_na.sex_of_driver.value_counts()
no_na.info()


# drop unknown values with 9 as weather
no_na.weather_conditions.value_counts()
no_na.weather_conditions = no_na.weather_conditions.replace(9,-1)
no_na.weather_conditions.value_counts()

# drop uknown values with 3 as urban or rural
no_na.urban_or_rural_area.value_counts()
no_na.urban_or_rural_area = no_na.urban_or_rural_area.replace(3,-1)
no_na.urban_or_rural_area.value_counts()


# drop all -1 values now
no_na =  no_na.loc[ ~no_na.isin([-1]).any(axis=1)]
no_na.reset_index(drop=True, inplace=True) # reset index


# no_na.to_csv("./data/casualties_no_negatives.csv", index=False)


# Formatting Column Datatypes ----------

df = no_na.copy()

# cat_cols = [
# "sex_of_casualty",
# "age_band_of_casualty",
# "casualty_severity",
# "casualty_type",
# "casualty_home_area_type",
# "police_force",
# "accident_severity",
# "day_of_week",
# "time",
# "local_authority_district_",
# "local_authority_highway_",
# "road_type",
# "junction_detail",
# "light_conditions",
# "weather_conditions",
# "urban_or_rural_area",
# "did_police_officer_attend_scene_of_accident",
# "vehicle_type",
# "vehicle_manoeuvre",
# "junction_location",
# "skidding_and_overturning",
# "1st_point_of_impact",
# "sex_of_driver",
# "age_of_driver",
# "age_band_of_driver",
# "driver_imd_decile",
# "driver_home_area_type"
# ]

# prep_cats = df.copy()
# prep_cats[cat_cols] = df[cat_cols].astype("category")
# prep_cats.info()



## Feature Engineering -------------------------------------------------------------------------
"""
look at:
    time - AM/PM
    casualty type
    casualty home area tyoe - bring down to two
    long-lat grouping
    number of vehicles - to bands?
    number of casualtyies - to bands
    first road class
"""

no_na_hist = no_na.hist(figsize=(40,40))

## Create Balanced Samples ----------------------------------------------------------------------------

np.random.seed(1)
# taking balanced number of each class of target variable

# df = prep_cats.copy()
df = no_na.copy()

# casualty severity
sample_variable = "casualty_severity"
size = df[sample_variable].value_counts().min()        # sample size
replace = True  # with replacement
fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]

cas = df.groupby(sample_variable, as_index=False).apply(fn)
cas.reset_index(drop=True, inplace=True) # reset index

df.info()

# sample_variable = "accident_severity"
# size = df[sample_variable].value_counts().min()        # sample size


# acc = df.groupby(sample_variable, as_index=False).apply(fn)
# acc.reset_index(drop=True, inplace=True) # reset index
# acc_hists = acc.hist(figsize=(40,40))

cas_hists = cas.hist(figsize=(40,40))




# Explore ------------------------------------------------------------------------------------


initial_hists = df.hist(figsize=(40,40))
initial_hists

df.columns
col_headers = df.columns

# # getting value counts
# for col in df.columns:
#     print(df[col].value_counts())

# # for severities
# for col in df.columns:
#     if "severity" in col:
#         print(df[col].value_counts())



## FEATURE ENGINEERING
df.info()

# sample_df = df.copy()
sample_df = cas.copy()



# 0. sex_of_casualty------
plot_var = "sex_of_casualty"
plt = sample_df[plot_var].value_counts().plot(kind="bar", title=plot_var.title())

# 1.age_of_casualty ------
plot_var = "age_of_casualty"
plt = df[plot_var].hist()
sample_df[plot_var].value_counts()

""" Complete - no changes """

# 2.age_band_of_casualty -------
plot_var = "age_band_of_casualty"
plt = sample_df[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
sample_df[plot_var].value_counts()

""" 
under 15 = 1-3
over 66 = 10-11
"""
# under 15s
df_age_bands= sample_df.copy() 
df_age_bands["age_band_of_casualty"] = np.where(df_age_bands["age_band_of_casualty"] <= 3 , 1, df_age_bands["age_band_of_casualty"])

# over 66
df_age_bands["age_band_of_casualty"] = np.where(df_age_bands["age_band_of_casualty"] >= 10 , 10, df_age_bands["age_band_of_casualty"])

df_age_bands["age_band_of_casualty"].value_counts().plot(kind="bar", title=plot_var.title())

""" Complete - added two bands """


## 4.casualty_type ------
plot_var = "casualty_type"
plt = df_age_bands[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_age_bands[plot_var].value_counts()

df_cas_type= df_age_bands.copy() 
df_cas_type["casualty_type"] = np.where(df_cas_type["casualty_type"] == 9 , 1, 
                                        df_cas_type["casualty_type"])

df_cas_type["casualty_type"] = np.where(df_cas_type["casualty_type"] == 0 , 2, df_cas_type["casualty_type"])
df_cas_type["casualty_type"] = np.where(df_cas_type["casualty_type"] > 2, 3, df_cas_type["casualty_type"])
                          
plt = df_cas_type[plot_var].value_counts().plot(kind="bar", title=plot_var.title())

""" COMPLETE - binned as Car occupant, pedestrian, other"""



## 5.casualty_home_area_type ------
df_cas_ha = df_cas_type.copy()
plot_var = "casualty_home_area_type"
plt = df_cas_ha[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_cas_ha[plot_var].value_counts()

""" Complete - leaving as they are for now, maybe balance into binary?"""


# 6. longitude and 7. latitude
df_longlat = df_cas_ha.copy()


### PLOTTING CASUALTIES


df_longlat.plot(kind="scatter", x="longitude", y = "latitude", cmap="jet", colorbar = False, alpha=0.09, figsize=(16,16), 
        )


### CREATE GROUPED LONGLATS
df_longlat["adj_long"] = df_longlat["longitude"].round(0)
df_longlat["adj_lat"] = df_longlat["latitude"].round(0)
 

df_longlat.plot(kind="scatter", x="adj_long", y = "adj_lat", cmap="jet", colorbar = False, alpha=0.09, figsize=(16,16), 
        )

# summarise count by long-lat region
df_longlat["accident_count_in_grouped_geos"] = df_longlat["sex_of_casualty"].groupby([df_longlat["adj_long"],df_longlat["adj_lat"]] ).transform("count")

# group adjusted longlat records with count
grouped_longlats = df_longlat.groupby(["adj_long", "adj_lat", "accident_count_in_grouped_geos"]).agg({"casualty_severity" : "mean"})
grouped_longlats.columns = ["average_cas_severity"]
grouped_longlats = grouped_longlats.reset_index()

print(grouped_longlats.sort_values(by="accident_count_in_grouped_geos", ascending=False).head(60))

# view using accident severity and grouped bins
plt = grouped_longlats.plot(kind="scatter", x="adj_long", y = "adj_lat", figsize=(12,12),
                   s=(grouped_longlats["accident_count_in_grouped_geos"]*0.8),
                   c=grouped_longlats["average_cas_severity"], cmap="jet_r", colorbar=True
        )
plt.set_title("Accidents by Coordinate Region", fontsize=24)
plt.set_xlabel("Longitude", fontsize=16)
plt.set_ylabel("Latitude", fontsize=16)



df_longlat = df_longlat.drop("accident_count_in_grouped_geos", axis=1)

""" 
MAY REVISIT - Created Grouped Longitude and Latitude Coordinates - rounded to 0 decimals

"""

# 8. police_force

df_policeF = df_longlat.copy()
plot_var = "police_force"
plt = df_policeF[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_policeF[plot_var].hist()
df_policeF[plot_var].value_counts()


""" REVISIT - CONSIDER DROPPING"""

# 9. accident_severity
df_accSev = df_policeF.copy()

plot_var = "accident_severity"
plt = df_accSev[plot_var].value_counts().plot(kind="bar", title=plot_var.title())

""" Keep as it is"""



# 10. number_of_vehicles
df_numVeh = df_accSev.copy()

plot_var = "number_of_vehicles"
plt = df_numVeh["number_of_vehicles"].value_counts().plot(kind="bar", title=plot_var.title())


df_numVeh["number_of_vehicles_bin"] = np.where(df_numVeh["number_of_vehicles"] >= 3 , "3+", df_numVeh["number_of_vehicles"].astype(str))
df_numVeh["number_of_vehicles_bin"].value_counts()

plt = df_numVeh["number_of_vehicles_bin"].value_counts().plot(kind="bar", title=plot_var.title())
""" """
# 11. number_of_casualties
df_numCas = df_numVeh.copy()

plot_var = "number_of_casualties"
plt = df_numCas[plot_var].value_counts().plot(kind="bar", title=plot_var.title())


df_numCas["number_of_casualties_bin"] = np.where(df_numCas["number_of_casualties"] >= 3 , "3+", df_numCas["number_of_casualties"].astype(str))
df_numCas["number_of_casualties_bin"].value_counts()

plt = df_numCas["number_of_casualties_bin"].value_counts().plot(kind="bar", title=plot_var.title())

""" Complete, binned into 3 categories - possibly come back and do quartiles?? """

# 12 Date

df_dates = df_numCas.copy()
df_dates["month"] = pd.DatetimeIndex(df_dates["date"]).month
df_dates["quarter"] = pd.DatetimeIndex(df_dates["date"]).quarter

plt = df_dates["month"].value_counts().plot(kind="bar", title=plot_var.title())
plt = df_dates["quarter"].value_counts().plot(kind="bar", title=plot_var.title())

df_dates.month.value_counts()
df_dates.quarter.value_counts()


""" Might come back and remove months or quarter - need to drop date"""

# 13. day_of_week
df_weekday = df_dates.copy()

plot_var = "day_of_week"
plt = df_weekday[plot_var].value_counts().plot(kind="bar", title=plot_var.title())

""" Keeping as is """


# 14. time

# create proxy categorical bins



df_time = df_weekday.copy()
df_time.time.value_counts()

df_time["time"] = pd.to_datetime(df_time["time"])
df_time['hour'] = pd.to_datetime(df_time['time'], format='%H:%M').dt.hour.astype("int64")

df_time.hour.value_counts()


conditions = [
    (df_time.hour  < 7 ),
    (df_time.hour  >= 7) & (df_time.hour  < 12),
    (df_time.hour  >= 12) & (df_time.hour  < 17),
    (df_time.hour  >= 17) & (df_time.hour  < 20),
    (df_time.hour  >= 20)
]

values = [1,2,3,4,5]

df_time["time_bin"] = np.select(conditions, values)

df_time["time_bin"].value_counts()


# df_ec["engine_capacity_group"] = pd.qcut(df_ec.engine_capacity_cc_, q=4) - quantiles for time?

plot_var = "time_bin"
df_time[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_time.hour.value_counts()


""" Grouped into 5 time bins based on hours - will come back and drop time"""


# 15 local_authority_district_
col_headers = df_time.columns

df_lad = df_time.copy()

plot_var = "local_authority_district_"
df_lad[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_lad[plot_var].value_counts()
df_lad[plot_var].nunique()

""" THINK WILL DROP THESE """


# 16 local_authority_highway_

df_lad = df_time.copy()

plot_var = "local_authority_highway_"

df_lad["lad_code"] = df_lad[plot_var].str.slice(0, 3)
df_lad.lad_code.value_counts()


""" Consider dropping out codes not in england """


# 17 road_type

df_roadtype = df_lad.copy()

plot_var = "road_type"

df_roadtype[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_roadtype[plot_var].value_counts()
df_roadtype[plot_var].nunique()

df_roadtype["single_carriageway"] = np.where(df_roadtype.road_type == 6, 1,0)
df_roadtype["single_carriageway"].value_counts()
df_roadtype["single_carriageway"].value_counts().plot(kind="bar", title=plot_var.title())

df_roadtype["single_carriageway"].nunique()


""" Made binary - will come back and remove original column"""


# 18. speed_limit

df_sl= df_roadtype.copy()

plot_var = "speed_limit"

df_sl[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_sl[plot_var].value_counts()
df_sl[plot_var].nunique()


""" Going to try keeping speed limits as they are  might come back and bin these into under or over 30 """


# 19. junction_detail

df_jd= df_sl.copy()

plot_var = "junction_detail"

df_jd[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_jd[plot_var].value_counts()
df_jd[plot_var].nunique()


""" THINK IL DROP THIS """



# 20. light_conditions

df_lc= df_jd.copy()

plot_var = "light_conditions"

df_lc[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_lc[plot_var].value_counts()
df_lc[plot_var].nunique()


df_lc["daylight"] = np.where(df_lc.light_conditions == 1, 1, 0)
df_lc["daylight"].value_counts().plot(kind="bar", title=plot_var.title())

""" Might be redundant with time of day - but maybe not"""


# 21. weather_conditions

df_wc= df_lc.copy()

plot_var = "weather_conditions"

df_wc[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_wc[plot_var].value_counts()
df_wc[plot_var].nunique()


df_wc["good_weather"] = np.where(df_lc.weather_conditions== 1, 1, 0)
df_wc["good_weather"].value_counts().plot(kind="bar", title=plot_var.title())

""" binned into good or bad - come back and get rid of original column"""

# 22. road_surface_conditions

df_rsc= df_wc.copy()

plot_var = "road_surface_conditions"

df_rsc[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_rsc[plot_var].value_counts()
df_rsc[plot_var].nunique()


df_rsc["dry_road"] = np.where(df_lc.road_surface_conditions== 1, 1, 0)
df_rsc["dry_road"].value_counts().plot(kind="bar", title=plot_var.title())

""" binned into dry or not dry - come back and get rid of original column"""


# 23. urban_or_rural_area

df_ura= df_rsc.copy()

plot_var = "urban_or_rural_area"

df_ura[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_ura[plot_var].value_counts()
df_ura[plot_var].nunique()

""" Perfect as it is"""





# 24. did_police_officer_attend_scene_of_accident


df_dpa = df_ura.copy()

plot_var = "did_police_officer_attend_scene_of_accident"
df_dpa[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_dpa[plot_var].value_counts()
df_dpa[plot_var].nunique()

df_dpa["did_police_officer_attend_scene_of_accident"] = np.where(df_dpa["did_police_officer_attend_scene_of_accident"] == 1, 1 , 0)


plot_var = "did_police_officer_attend_scene_of_accident"
df_dpa[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_dpa[plot_var].value_counts()

""" Consider Dropping """



# 25 . lsoa_of_accident_location
df_lsao= df_dpa.copy()
plot_var = "lsoa_of_accident_location"
df_lsao[plot_var].value_counts()

df_lsao["lsoa_code"] = df_lsao[plot_var].str.slice(0, 3)
df_lsao.lsoa_code.value_counts()

df_lsao[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_lsao[plot_var].value_counts()
df_lsao[plot_var].nunique()

""" NOT INCLUDING - DROP"""



# 26 . vehicle_type
df_vt= df_dpa.copy()

plot_var = "vehicle_type"
df_vt[plot_var].value_counts()

df_vt[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_vt[plot_var].value_counts()
df_vt[plot_var].nunique()

"""
8, 9 = car/taxi
1 - 5, 22, 23, 97,  = bike
all others = other
"""


df_vt["vehicle_type"] = df_vt.vehicle_type.replace([1,2,3,4,5,22,23,97], 2)
df_vt["vehicle_type"] = df_vt.vehicle_type.replace([8,9], 1)
df_vt["vehicle_type"] = np.where(df_vt.vehicle_type > 2, 3, df_vt.vehicle_type)

df_vt[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_vt[plot_var].value_counts()
df_vt[plot_var].nunique()

df_vt.vehicle_type.value_counts()

""" Grouped into CARS, BIKES, OTHERS"""

# 27 . vehicle_manoeuvre
df_vm= df_vt.copy()

plot_var = "vehicle_manoeuvre"
df_vm[plot_var].value_counts()

df_vm[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_vm[plot_var].value_counts()
df_vm[plot_var].nunique()

""" DROPPING => doesnt seem useful """


# 28 . junction_location
df_jl= df_vm.copy()

plot_var = "junction_location"
df_jl[plot_var].value_counts()

df_jl[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_jl[plot_var].value_counts()
df_jl[plot_var].nunique()

df_jl["near_junction"] = np.where(df_jl[plot_var] == 0, 0, 1)

df_jl["near_junction"].value_counts().plot(kind="bar", title=plot_var.title())
df_jl["near_junction"].value_counts()
df_jl["near_junction"].nunique()

""" grouped to binary "near_junction" - will come back and remove original column """


# 29 skidding_and_overturning

col_headers = df_jl.columns

df_so = df_jl.copy()

plot_var = "skidding_and_overturning"
df_so[plot_var].value_counts()

df_so[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_so[plot_var].value_counts()
df_so[plot_var].nunique()

df_jl["skid_or_overturn"] = np.where(df_so[plot_var] == 0, 0, 1)

df_jl["skid_or_overturn"].value_counts().plot(kind="bar", title=plot_var.title())
df_jl["skid_or_overturn"].value_counts()
df_jl["skid_or_overturn"].nunique()

""" grouped to binary "skid_or_overturn" - will come back and remove original column """




# 30  1st_point_of_impact

df_1p = df_so.copy()

plot_var = "1st_point_of_impact"
df_1p[plot_var].value_counts()

df_1p[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_1p[plot_var].value_counts()
df_1p[plot_var].nunique()

""" keeping as is"""



# 31  sex_of_driver

df_sx = df_1p.copy()

plot_var = "sex_of_driver"
df_sx[plot_var].value_counts()

df_sx[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_sx[plot_var].value_counts()
df_sx[plot_var].nunique()

""" keeping as is - NEED TO REVISIT AND DROP VALUES with 3 - AT START of flow"""




# 32  age_of_driver

df_age = df_sx.copy()

plot_var = "age_of_driver"
df_age[plot_var].value_counts()

df_age[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_age[plot_var].value_counts()
df_age[plot_var].nunique()



df_age[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_age[plot_var].value_counts()
df_age[plot_var].nunique()

""" DROP CONTINOUS AGE COLUMN AND KEEP THE BAND"""


# 33  age_band_of_driver

df_ageband = df_age.copy()

plot_var = "age_band_of_driver"
df_ageband[plot_var].value_counts()

df_ageband[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_ageband[plot_var].value_counts()
df_ageband[plot_var].nunique()



df_ageband[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_ageband[plot_var].value_counts()
df_ageband[plot_var].nunique()

""" keeping as is """


# 34  engine_capacity_cc_

df_ec = df_ageband.copy()

plot_var = "engine_capacity_cc_"
df_ec[plot_var].value_counts()

df_ec[plot_var].plot(title=plot_var.title())
df_ec[plot_var].value_counts()
df_ec[plot_var].nunique()

df_ec.engine_capacity_cc_.describe()


df_ec["engine_capacity_group"] = pd.qcut(df_ec.engine_capacity_cc_, q=4, labels=["Q1","Q2","Q3","Q4"])

df_ec["engine_capacity_group"].value_counts().plot(kind="bar", title="engine_capacity_group")
df_ec["engine_capacity_group"].value_counts()
df_ec["engine_capacity_group"].nunique()

""" binned into 4 quantiles"""


# 35  age_of_vehicle

df_aov = df_ec.copy()

plot_var = "age_of_vehicle"
df_aov[plot_var].value_counts()

df_ec[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_aov[plot_var].value_counts()
df_aov[plot_var].nunique()


df_aov.age_of_vehicle.describe()


df_aov["age_band_of_vehicle"]= pd.qcut(df_aov.age_of_vehicle, q=8, labels=["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"])

df_aov["age_band_of_vehicle"].value_counts().plot(kind="bar", title="age_band_of_vehicle")

""" binned into 8 quantiles"""




# 36  driver_imd_decile

df_imdd= df_aov.copy()

plot_var = "driver_imd_decile"
df_imdd[plot_var].value_counts()

df_imdd[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_imdd[plot_var].value_counts()
df_imdd[plot_var].nunique()


""" keeping as is"""



# 37  driver_imd_decile
df_hat= df_imdd.copy()

plot_var = "driver_home_area_type"
df_hat[plot_var].value_counts()

df_hat[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_hat[plot_var].value_counts()
df_hat[plot_var].nunique()


""" keeping as is"""

# 38  driver_imd_decile

df_hat= df_imdd.copy()

plot_var = "driver_home_area_type"
df_hat[plot_var].value_counts()

df_hat[plot_var].value_counts().plot(kind="bar", title=plot_var.title())
df_hat[plot_var].value_counts()
df_hat[plot_var].nunique()

""" keeping as is"""








# # pickling processed data
# with open ("./data/pickles/prepared_df.pkl", "wb") as f:
#     pickle.dump(full_set, f)

# with open("./data/pickles/prepared_df.pkl", "rb") as f:
#     prepared_df = pickle.load(f)


# write prepared data to CSV

prepared_df = df_hat.copy()

# prepared_df.to_csv("./data/full_prepared_dataset.csv", index=False)

prepared_df.hist(figsize=(40,40))

## FORMAT DATA TYPES ------------------------------------------------------

prepared_df.age_band_of_driver.value_counts()

## FORMAT for MACHINE LEARNING ----------------------------------------------



drop_cols2 = [
"age_of_casualty",
"longitude",
"latitude",
"police_force",
"number_of_vehicles",
"date",
"quarter",
"time",
"local_authority_district_",
"local_authority_highway_",
"road_type",
"junction_detail",
"light_conditions",
"weather_conditions",
"road_surface_conditions",
"lsoa_of_accident_location",
"vehicle_manoeuvre",
"junction_location",
"skidding_and_overturning",
"age_of_driver",
"engine_capacity_cc_",
"age_of_vehicle",
"hour", 
"number_of_casualties"
]


drop_cols3 = [
"age_of_casualty",
"longitude",
"latitude",
"police_force",
"number_of_vehicles",
"date",
"quarter",
"time",
"local_authority_district_",
"local_authority_highway_",
"road_type",
"junction_detail",
"light_conditions",
"weather_conditions",
"road_surface_conditions",
"lsoa_of_accident_location",
"vehicle_manoeuvre",
"junction_location",
"skidding_and_overturning",
"age_of_driver",
"engine_capacity_cc_",
"age_of_vehicle",
"hour", 
"number_of_casualties"
]




ml_set = prepared_df.copy()
ml_set = ml_set.drop(drop_cols2, axis=1)

# ml_format = ml_set.copy()


# for col in ml_set.columns:
#     ml_set[col] = ml_set[col].astype("category")




# # pickling processed data
with open (f"./data/pickles/{outfile_name}.pkl", "wb") as f:
    pickle.dump(ml_set, f)



ml_set.to_csv(f"./data/{outfile_name}.csv", index=False)

# with open("./data/pickles/ml_set.pkl", "rb") as f:
#     prepared_df = pickle.load(f)











