

""" ####################  SET WORKING DIRECTOR TO DIRECTORY OF THIS SCRIPT #####################"""

project_path = "C:/Users/alanm/OneDrive - National College of Ireland/Final Year/Software Project/Project"


# ----------------------------------------------------------------------------------

""" ================================ SETUP =================================="""


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
# full_set = pd.read_csv("./data/ALL_RECORDS.csv")

# with open (f"./data/pickles/FULL_SET.pkl", "wb") as f:
#     pickle.dump(full_set, f)

with open(f"./data/pickles/FULL_SET.pkl", "rb") as f:
    full_set = pickle.load(f)




"""----------------------------------------------------------------------------"""

# ---------------------------------------------------------------------------------- #
""" ================================ Data Preparation ======================="""

def prepare_dataset(full_set):

    # format column headers
    full_set.columns = full_set.columns.str.replace("Vehicles.", "")
    full_set = full_set.clean_names()
    full_set.columns
    
    
    # Dropping Columns ---------------
    drop_cols =[
    "source_name", # identifiers
    "accident_index",
    "vehicle_reference",
    "casualty_reference",
    "location_easting_osgr", # using longlats
    "location_northing_osgr",
    "lsoa_of_accident_location", # too many missing values
    ]
    
    dropped = full_set.drop(drop_cols, axis=1)
    
    
    # Dropping Missing Values ---------------------------------------- #
    
    # drop records with any values as -1 (missing information)
    nas = dropped.copy()
    nas.isna().sum().sort_values(ascending=False)
    
    no_na = nas.fillna(-1)
    nas.isna().sum()
    
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
    #--------------------------------------------------------------- #
    
    
    # Feature Engineering -------------------------------------------#
    
    # invert severity
    sev = no_na.copy()
    sev.casualty_severity = np.where(sev.casualty_severity == 1, 0, sev.casualty_severity)
    sev.casualty_severity = np.where(sev.casualty_severity == 3, 1, sev.casualty_severity)
    sev.casualty_severity = np.where(sev.casualty_severity == 0, 3, sev.casualty_severity)
    
    # Create Binary Fatal or Not Fatal
    df_fatal = sev.copy()
    df_fatal["fatal"] = np.where(df_fatal.casualty_severity == 3,1,0)
    
    
    # Create month and hour columns
    df_time = df_fatal.copy()
    df_time["month"] = pd.DatetimeIndex(df_time["date"]).month
    df_time["time"] = pd.to_datetime(df_time["time"])
    df_time['hour'] = pd.to_datetime(df_time['time'], format='%H:%M').dt.hour.astype("int64")
    df_time = df_time.drop(["time", "date"], axis = 1)
    
    
    # Create Adjusted Longlats
    df_longlat = df_time.copy()
    df_longlat["adj_long"] = df_longlat["longitude"].round(0)
    df_longlat["adj_lat"] = df_longlat["latitude"].round(0)
    
    
    # # Grouping Local Area Highway Code
    # df_lah = df_longlat.copy()
    # df_lah["local_authority_code"] = df_lah["local_authority_highway_"].str.slice(0, 3)
    # df_lah.local_authority_code.value_counts()
    
    
    # df_prepped = df_lah.copy()
    
    df_prepped = df_longlat.copy()
    
    return df_prepped



# with open (f"./data/pickles/MAY_Prepped.pkl", "wb") as f:
#     pickle.dump(df_prepped, f)

# with open(f"./data/pickles/MAY_Prepped.pkl", "rb") as f:
#     df_prepped = pickle.load(f)


# Function to exclude columns from a dataset
def exclude_columns(df, excluded):
    dropped = df.drop(excluded, axis=1)
    return dropped


# -----------------------------------------------------------------------#

""" ============================= DATA PARTITIONING ============================"""

def partition_dataset(dataset, target):
        
    # Generate Stratified Splits
    train, test = strat_split(dataset, target)
    
    # Split into predictor and target
    X_train, y_train = split_set(train, target)
    X_test, y_test = split_set(test, target)
    
    return X_train, X_test, y_train, y_test




""" ============= FEATURE SELECTION FUNCTIONS - USING CHI-SQUARE TESTS ============"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.model_selection import StratifiedShuffleSplit




# stratified split
def strat_split(df, target_var):
    split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 1)
    
    for train_index, test_index in split.split(df, df[target_var]):
        train = df.loc[train_index]
        test = df.loc[test_index]
        
        return train, test

# load the dataset
def split_set(data, target_var):
	X = data.drop(target_var, axis=1)
	y = data[target_var].to_frame()
	X = X.astype(str)
	return X, y

# prepare input data
def prepare_predictors(X_train, X_test):
    oe = OrdinalEncoder()
    X_train_encoded = oe.fit_transform(X_train)
    X_test_encoded = oe.fit_transform(X_test)
    return X_train_encoded, X_test_encoded
 
# prepare target variables
def prepare_target(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# select features
def select_features(X_train, y_train, X_test, k ):
	fs = SelectKBest(score_func=chi2, k=k)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs


# Create Feature Selected Train and Tests
def generate_feature_selections(X_train, X_test, y_train, y_test, k):
    X_train_encoded, X_test_encoded = prepare_predictors(X_train, X_test)
    y_train_encoded, y_test_encoded = prepare_target(y_train.values.ravel(), y_test.values.ravel())
    X_train_fs, X_test_fs, fs = select_features(X_train_encoded, y_train_encoded, X_test_encoded, k)
    

    
    fs_scoreboard = list()
    for i in range(len(fs.scores_)):
        print(f'Feature {i}, {X_train.columns[i]}, {fs.get_support()[i]}, {fs.scores_[i]}')
        fs_scoreboard.append((i, X_train.columns[i], fs.get_support()[i], fs.scores_[i]))
    
    fs_scoreboard = pd.DataFrame(fs_scoreboard)
    fs_scoreboard.columns = ["Feature No.", "Feature Name", "SELECTED", "Feature Score"]
    
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    
    selected = fs_scoreboard["Feature Name"].loc[fs_scoreboard["SELECTED"] == True].tolist()
    
    X_train_fs = X_train[selected]
    X_test_fs = X_test[selected]
    
    return X_train_fs, X_test_fs, y_train, y_test, fs_scoreboard



""" ------------------------------------------------------------------------------------------------------"""



"""=========================== PREDICTIVE MODELS  ========================="""




def run_models(test_name, X_train, X_test, y_train, y_test):
    
    np.random.seed(1)

    """ DECISION TREE """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    
    dt_clf = DecisionTreeClassifier(random_state=1)
    
    dt_clf.fit(X_train, y_train)
    
    # get score for model
    dt_predictions = dt_clf.predict(X_test)
    dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
    
    dt_score = metrics.accuracy_score(y_test, dt_predictions)*100
    
    print("Decision Tree Accuracy:",dt_score, "Confusion Matrix:", dt_confusion_matrix, sep=("\n"))
    
    """ RANDOM FOREST """
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state = 1)
    rf_classifier.fit(X_train, y_train.values.ravel())
    
    # get score for model
    rf_predictions = rf_classifier.predict(X_test)
    
    rf_score = rf_classifier.score(X_test, y_test) * 100
    rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
    plot_confusion_matrix(rf_classifier, X_test, y_test) 
    
    print("Random Forest Accuracy:", rf_score, "Confusion Matrix:", rf_confusion_matrix, sep=("\n"))
    
    
    
    """ GAUSSIAN NAIVE BAYES """
    from sklearn.naive_bayes import GaussianNB
    
    gnb_classifier = GaussianNB()
    gnb_classifier.fit(X_train, y_train.values.ravel())
    
    # get score for model
    gnb_predictions = gnb_classifier.predict(X_test)
    gnb_score = gnb_classifier.score(X_test, y_test) * 100
    
    gnb_confusion_matrix = confusion_matrix(y_test, gnb_predictions)
    print("Gaussian Naive Bayes Accuracy", gnb_score,"Confusion Matrix:", gnb_confusion_matrix, sep=("\n"))
    
    
    
    """ NEURAL NETWORK MLP Classifier """
    from sklearn.neural_network  import MLPClassifier
    
    mlp_classifier = MLPClassifier(random_state=1)
    mlp_classifier.fit(X_train, y_train.values.ravel())
    
    # get score for model
    mlp_predictions = mlp_classifier.predict(X_test)
    mlp_score = mlp_classifier.score(X_test, y_test) * 100
    
    # confusion matrix
    mlp_confusion_matrix = confusion_matrix(y_test, mlp_predictions)
    print("Neural Network Accuracy:", mlp_score, "Confusion Matrix", mlp_confusion_matrix, sep=("\n"))
    
    
    """ GRADIENT BOOSTING MACHINE """
    from sklearn.ensemble import GradientBoostingClassifier
    
    gbc_classifier = GradientBoostingClassifier(random_state=1)
    
    # gbc_classifier = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=100, random_state=1)
    
    gbc_classifier.fit(X_train, y_train.values.ravel())
    
    # get score for model
    gbc_predictions = gbc_classifier.predict(X_test)
    
    gbc_score = gbc_classifier.score(X_test, y_test)*100
    
    gbc_confusion_matrix = confusion_matrix(y_test, gbc_predictions)
    print("Gradient Boosting Machine:", gbc_score, "Confusion Matrix:", gbc_confusion_matrix, sep=("\n"))




    # ------------------ RESULTS ---------------#
    
    
    
    data = {'Dataset': test_name ,
            'Model': 
                ['Decision Tree', 
                 'Random Forest', 
                 'Gaussian Naive Bayes',
                 'MLP Neural Net Classifier',
                 'Gradient Boosting Machine'
                 ], 
                'Score': [dt_score, rf_score, gnb_score, mlp_score, gbc_score],
                'Confusion Matrix' : 
                    [dt_confusion_matrix, 
                     rf_confusion_matrix, 
                     gnb_confusion_matrix, 
                     mlp_confusion_matrix, 
                     gbc_confusion_matrix],
                    "Features Used": len(pd.DataFrame(X_train).columns)
                    }  
      
    results_table = pd.DataFrame(data)
    
#-----------------------------------------------#
    
    return results_table



    


"""------------------------------------------------------------------------------------------------------"""

# PRINT RESULTS FUNCTION
def save_results(test_name, results, fscoreboard=None):
    results.to_csv(f"./results/{test_name}_results.csv")
    if fscoreboard:
        results.to_csv(f"./results/{test_name}_scoreboard.csv")


""" ===================== CREATE BALANCED SAMPLE SET =========================="""

# Create Balanced Sample -------------------------------------- #

def create_balanced_dataset(dataset, target):
      
    np.random.seed(1)
    
    size = dataset[target].value_counts().min()
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, True),:]
    
    balanced_set = dataset.groupby(target, as_index=False).apply(fn)
    balanced_set.reset_index(drop=True, inplace=True) # reset index
    
    balanced_set.info()
    balanced_set[target].value_counts()
    return balanced_set

""" -------------------------------------------------------------------"""






""" =================================== RUNNING MODELS AND GENERATING RESULTS ====================================="""

def run_analysis(analysis_name, dataset, target, fs = False, k = 0):

    X_train, X_test, y_train, y_test = partition_dataset(dataset, target)
        
    results = run_models(analysis_name+"_results", X_train, X_test, y_train, y_test)
    save_results(analysis_name, results)
    
    
def run_feature_analysis(analysis_name, dataset, target, fs = False, k = 0):

    X_train, X_test, y_train, y_test = partition_dataset(dataset, target)
    X_train_fs, X_test_fs, y_train_fs, y_test_fs, fscoreboard = generate_feature_selections(X_train, X_test, y_train, y_test, k)
        
    results = run_models(analysis_name+"_results", X_train_fs, X_test_fs, y_train_fs, y_test_fs)
    
    
    
    return results, fscoreboard
    
    


# SELECT FEATURES
excluded_cols = ["longitude", "latitude", "accident_severity", "local_authority_highway_", "casualty_severity"]



# CREATE DATASETS 
prepared_dataset = prepare_dataset(full_set)
prepared_dataset = exclude_columns(prepared_dataset, excluded_cols)
balanced_dataset = create_balanced_dataset(prepared_dataset, "fatal")

    
run_analysis("TEST balanced_sample_all_features_fatal_defaultsettings", balanced_dataset, "fatal")

run_analysis("prepared_dataset_all_features_fatal_defaultsettings", prepared_dataset, "fatal")






# RUN MODELS ON BALANCED SET
run_analysis("BALANCED_SET_DEFAULT_VALUES_FATAL", balanced_dataset, "fatal", fs=True, k="all" )


# RUN MODELS ON FULL SET (longer processing time of course)
# run_analysis("PREPARED_SET_DEFAULT_VALUES_FATAL", prepared_dataset, "fatal", fs=True, k="all" )




# RUN MODELS WITH BEST FEATURE SELECTIONS BASED ON CHI_SQUARED TESTS (looping through all)

# i = 57
# counter = 1

# feature_analysis_results = pd.DataFrame()
# while i > 0:
#     print(f"--------------ITERATION #{counter}-----------")
#     counter=counter+1
#     title = f"feature_selection_analysis_balanced_set_{i}"
#     results, scoreboard = run_feature_analysis(title, balanced_dataset, "fatal", fs=True, k=i)
#     feature_analysis_results = feature_analysis_results.append(results)
#     i = i-1
    
# save_results("FEATURE_SELECTION_ANALYSIS_BALANCED_SET",feature_analysis_results)




# THE BELOW PROCESS REQUIRES HUGE PROCESSING TIME

# i = 50
# counter = 1

# feature_analysis_results_prepared_set = pd.DataFrame()

# X_train, X_test, y_train, y_test = partition_dataset(prepared_dataset, "fatal")

# while i >= 0:
#     print(f"--------------ITERATION #{counter}-----------")
#     counter=counter+1
#     title = f"feature_selection_analysis_full_prepared_set_{i}"
#     results = run_feature_analysis("FEATURE_SELECTION_ANALYSIS_FULL_DATASET", prepared_dataset, "fatal", fs=True, k=i)
    
#     X_train_fs, X_test_fs, y_train_fs, y_test_fs, fscoreboard = generate_feature_selections(X_train, X_test, y_train, y_test, k=i)
        
#     results = run_models("FEATURE_SELECTION_ANALYSIS_FULL_DATASET_results", X_train_fs, X_test_fs, y_train_fs, y_test_fs)
    
#     feature_analysis_results_prepared_set = feature_analysis_results_prepared_set.append(results)
#     i = i-10
    
    
    
    
    
# save_results("FEATURE_SELECTION_ANALYSIS_FULL_DATASET",feature_analysis_results_prepared_set)


""" ================================== VISUALISATIONS ==================================================="""


combos = prepared_dataset.groupby(["adj_long", "adj_lat"]).size().reset_index()


vis = full_set.copy()


# format column headers
vis.columns = vis.columns.str.replace("Vehicles.", "")
vis = vis.clean_names()
vis.columns

# drop all -1 values now
vis = vis[vis['date'].notna()]
vis = vis[vis['longitude'].notna()]
vis = vis[vis['latitude'].notna()]

 # invert severity

vis.casualty_severity = np.where(vis.casualty_severity == 1, 0, vis.casualty_severity)
vis.casualty_severity = np.where(vis.casualty_severity == 3, 1, vis.casualty_severity)
vis.casualty_severity = np.where(vis.casualty_severity == 0, 3, vis.casualty_severity)

# Create Binary Fatal or Not Fatal

vis["fatal"] = np.where(vis.casualty_severity == 3,1,0)


# Create month and hour columns

vis[vis.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]


vis["month"] = pd.DatetimeIndex(vis["date"]).month
vis["year"] = pd.DatetimeIndex(vis["date"]).year


vis["time"] = vis["time"].fillna(-1)
vis = vis[vis.time != -1]

vis["time"] = pd.to_datetime(vis["time"])


    
vis["hour"] = pd.to_datetime(vis["time"], format='%H:%M').dt.hour.astype("int64")



import seaborn as sns


var = vis["hour"]
f, ax = plt.subplots(figsize=(6,6), nrows=2,)
sns.countplot(x="year", data = vis, ax=ax[0])
s.countplot(x="month", data = vis, ax=ax[1])
f, ax = plt.subplots(figsize=(12,12), nrows=2,)
sns.countplot(x="day_of_week", data = vis, ax=ax[0])
sns.countplot(x="hour", data = vis, ax=ax[1])


# plt.savefig("./plots/pol_countplot.png")
plt.show()




f, ax = plt.subplots(figsize=(6,6), nrows=2,)
sns.countplot(x="year", data = vis, ax=ax[0])
s.countplot(x="month", data = vis, ax=ax[1])
f, ax = plt.subplots(figsize=(12,12), nrows=2,)
sns.countplot(x="day_of_week", data = vis, ax=ax[0])
sns.countplot(x="hour", data = vis, ax=ax[1])


# plt.savefig("./plots/pol_countplot.png")
plt.show()





























