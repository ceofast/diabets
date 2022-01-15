# Feature Engineering

# Bussiness Problem

# It is desired to develop a machine learning model that can predict whether people
# have diabetes when their characteristics are specified. You are expected to perform
# the necessary data analysis and feature engineering steps before developing the model.

# Dataset Story

# The dataset is part of the large dataset held at the National Institutes of
# Diabetes-Digestive-Kidney Diseases in the USA. Data used for diabetes research
# on Pima Indian women aged 21 and over living in Phoenix, the 5th largest city
# of the State of Arizona in the USA.
# The target variable is specified as "outcome"; 1 indicates positive diabetes
# test result, 0 indicates negative.

# Pregnancies : Number of pregnancies
# Glucose : 2-hour plasma glucose concentration in the oral glucose tolerance test
# Blood Pressure : Blood Pressure (Low blood pressure) (mmHg)
# SkinThickness : Skin Thickness
# Insulin : 2-hour serum insulin (mu U/ml)
# DiabetesPedigreeFunction : Function (2 hour plasma glucose concentration in oral glucose tolerance test)
# BMI : Body Mass Index
# Age : Age(year)
# Outcome : Have the disease (1) or not (0)

# Project Tasks

# Step 1 : Examine the overall picture.
# Step 2 : Capture the numeric and categorical variables.
# Step 3 :  Analyze the numerical and categorical variables.
# Step 4 : Perform target variable analysis. (The mean of the target variable according
# to the categorical variables, the mean of the numeric variables according to the target variable)
# Step 5 : Analyze the outlier observation.
# Step 6 : Perform missing observation analysis.
# Step 7 : Perform correlation analysis.

# Task 2 : Feature Engineering

# Step 1 : Take necessary actions for missing and outlier values. There are no missing observations in the data set,
# but Glucose, Insulin etc. Observation units containing 0 in the variables may represent the missing value.
# E.g; a person's glucose or insulin value will not be 0. Considering this situation, you can assign the zero values
# to the relevant values as NaN and then apply the operations to the missing values.
# Step 2 : Create new variables.
# Step 3 : Perform the encoding operations.
# Step 4 : Standardize for numeric variables.
# Step 5 : Create the model.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import warnings
warnings.simplefilter("ignore")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def load_df():
    data = pd.read_csv(r"/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_6/datasets/diabetes.csv")
    return data

df_ = load_df()
df = df_.copy()
df.head()
#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin      BMI  DiabetesPedigreeFunction  Age  Outcome
# 0            6      148             72             35        0 33.60000                   0.62700   50        1
# 1            1       85             66             29        0 26.60000                   0.35100   31        0
# 2            8      183             64              0        0 23.30000                   0.67200   32        1
# 3            1       89             66             23       94 28.10000                   0.16700   21        0
# 4            0      137             40             35      168 43.10000                   2.28800   33        1

def check_df(dataframe, head=5, tail = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head ######################")
    print(dataframe.head(head))
    print("##################### Tail ######################")
    print(dataframe.tail(tail))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# ##################### Shape #####################
# (768, 9)
# ##################### Types #####################
# Pregnancies                   int64
# Glucose                       int64
# BloodPressure                 int64
# SkinThickness                 int64
# Insulin                       int64
# BMI                         float64
# DiabetesPedigreeFunction    float64
# Age                           int64
# Outcome                       int64
# dtype: object
# ##################### Head ######################
#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin      BMI  DiabetesPedigreeFunction  Age  Outcome
# 0            6      148             72             35        0 33.60000                   0.62700   50        1
# 1            1       85             66             29        0 26.60000                   0.35100   31        0
# 2            8      183             64              0        0 23.30000                   0.67200   32        1
# 3            1       89             66             23       94 28.10000                   0.16700   21        0
# 4            0      137             40             35      168 43.10000                   2.28800   33        1
# ##################### Tail ######################
#      Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin      BMI  DiabetesPedigreeFunction  Age  Outcome
# 763           10      101             76             48      180 32.90000                   0.17100   63        0
# 764            2      122             70             27        0 36.80000                   0.34000   27        0
# 765            5      121             72             23      112 26.20000                   0.24500   30        0
# 766            1      126             60              0        0 30.10000                   0.34900   47        1
# 767            1       93             70             31        0 30.40000                   0.31500   23        0
# ##################### NA ########################
# Pregnancies                 0
# Glucose                     0
# BloodPressure               0
# SkinThickness               0
# Insulin                     0
# BMI                         0
# DiabetesPedigreeFunction    0
# Age                         0
# Outcome                     0
# dtype: int64
# ##################### Quantiles #####################
#                           0.00000  0.05000   0.50000   0.95000   0.99000   1.00000
# Pregnancies               0.00000  0.00000   3.00000  10.00000  13.00000  17.00000
# Glucose                   0.00000 79.00000 117.00000 181.00000 196.00000 199.00000
# BloodPressure             0.00000 38.70000  72.00000  90.00000 106.00000 122.00000
# SkinThickness             0.00000  0.00000  23.00000  44.00000  51.33000  99.00000
# Insulin                   0.00000  0.00000  30.50000 293.00000 519.90000 846.00000
# BMI                       0.00000 21.80000  32.00000  44.39500  50.75900  67.10000
# DiabetesPedigreeFunction  0.07800  0.14035   0.37250   1.13285   1.69833   2.42000
# Age                      21.00000 21.00000  29.00000  58.00000  67.00000  81.00000
# Outcome                   0.00000  0.00000   0.00000   1.00000   1.00000   1.00000

df.info()
# #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Pregnancies               768 non-null    int64
#  1   Glucose                   768 non-null    int64
#  2   BloodPressure             768 non-null    int64
#  3   SkinThickness             768 non-null    int64
#  4   Insulin                   768 non-null    int64
#  5   BMI                       768 non-null    float64
#  6   DiabetesPedigreeFunction  768 non-null    float64
#  7   Age                       768 non-null    int64
#  8   Outcome                   768 non-null    int64

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# bservations: 768
# Variables: 9
# cat_cols: 1
# num_cols: 8
# cat_but_car: 0
# num_but_cat: 1

cat_cols
# ['Outcome']

num_cols
# ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

df[num_cols].describe().T
#                              count      mean       std      min      25%       50%       75%       max
# Pregnancies              768.00000   3.84505   3.36958  0.00000  1.00000   3.00000   6.00000  17.00000
# Glucose                  768.00000 120.89453  31.97262  0.00000 99.00000 117.00000 140.25000 199.00000
# BloodPressure            768.00000  69.10547  19.35581  0.00000 62.00000  72.00000  80.00000 122.00000
# SkinThickness            768.00000  20.53646  15.95222  0.00000  0.00000  23.00000  32.00000  99.00000
# Insulin                  768.00000  79.79948 115.24400  0.00000  0.00000  30.50000 127.25000 846.00000
# BMI                      768.00000  31.99258   7.88416  0.00000 27.30000  32.00000  36.60000  67.10000
# DiabetesPedigreeFunction 768.00000   0.47188   0.33133  0.07800  0.24375   0.37250   0.62625   2.42000
# Age                      768.00000  33.24089  11.76023 21.00000 24.00000  29.00000  41.00000  81.00000

df[cat_cols].describe()
#        Outcome
# count 768.00000
# mean    0.34896
# std     0.47695
# min     0.00000
# 25%     0.00000
# 50%     0.00000
# 75%     1.00000
# max     1.00000

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)
# # (Pregnancies                  -6.50000
# # Glucose                      37.12500
# # BloodPressure                35.00000
# # SkinThickness               -48.00000
# # Insulin                    -190.87500
# # BMI                          13.35000
# # DiabetesPedigreeFunction     -0.33000
# # Age                          -1.50000
# # dtype: float64, Pregnancies                 13.50000
# # Glucose                    202.12500
# # BloodPressure              107.00000
# # SkinThickness               80.00000
# # Insulin                    318.12500
# # BMI                         50.55000
# # DiabetesPedigreeFunction     1.20000
# # Age                         66.50000

outlier_thresholds(df, cat_cols)
# (Outcome   -1.50000
# dtype: float64, Outcome   2.50000
# dtype: float64)

check_outlier(df, num_cols)
# True

check_outlier(df, cat_cols)
# False

for col in num_cols:
    print(col, check_outlier(df, col))
# Pregnancies True
# Glucose True
# BloodPressure True
# SkinThickness True
# Insulin True
# BMI True
# DiabetesPedigreeFunction True
# Age True

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, num_cols)
#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin  BMI  DiabetesPedigreeFunction  Age  Outcome
# 0          NaN      NaN            NaN            NaN      NaN  NaN                       NaN  NaN      NaN
# 1          NaN      NaN            NaN            NaN      NaN  NaN                       NaN  NaN      NaN
# 2          NaN      NaN            NaN            NaN      NaN  NaN                       NaN  NaN      NaN
# 3          NaN      NaN            NaN            NaN      NaN  NaN                       NaN  NaN      NaN
# 4          NaN      NaN            NaN            NaN      NaN  NaN                   2.28800  NaN      NaN

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in df.columns:
    target_summary_with_num(df, "Outcome", col)
#          Pregnancies
# Outcome
# 0            3.29800
# 1            4.86567
#           Glucose
# Outcome
# 0       109.98000
# 1       141.25746
#          BloodPressure
# Outcome
# 0             68.18400
# 1             70.82463
#          SkinThickness
# Outcome
# 0             19.66400
# 1             22.16418
#           Insulin
# Outcome
# 0        68.79200
# 1       100.33582
#              BMI
# Outcome
# 0       30.30420
# 1       35.14254
#          DiabetesPedigreeFunction
# Outcome
# 0                         0.42973
# 1                         0.55050
#              Age
# Outcome
# 0       31.19000
# 1       37.06716
#          Outcome
# Outcome
# 0        0.00000
# 1        1.00000

cor = df.corr(method="pearson")
cor
#                           Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin     BMI  DiabetesPedigreeFunction      Age  Outcome
# Pregnancies                   1.00000  0.12946        0.14128       -0.08167 -0.07353 0.01768                  -0.03352  0.54434  0.22190
# Glucose                       0.12946  1.00000        0.15259        0.05733  0.33136 0.22107                   0.13734  0.26351  0.46658
# BloodPressure                 0.14128  0.15259        1.00000        0.20737  0.08893 0.28181                   0.04126  0.23953  0.06507
# SkinThickness                -0.08167  0.05733        0.20737        1.00000  0.43678 0.39257                   0.18393 -0.11397  0.07475
# Insulin                      -0.07353  0.33136        0.08893        0.43678  1.00000 0.19786                   0.18507 -0.04216  0.13055
# BMI                           0.01768  0.22107        0.28181        0.39257  0.19786 1.00000                   0.14065  0.03624  0.29269
# DiabetesPedigreeFunction     -0.03352  0.13734        0.04126        0.18393  0.18507 0.14065                   1.00000  0.03356  0.17384
# Age                           0.54434  0.26351        0.23953       -0.11397 -0.04216 0.03624                   0.03356  1.00000  0.23836
# Outcome                       0.22190  0.46658        0.06507        0.07475  0.13055 0.29269                   0.17384  0.23836  1.00000

sns.heatmap(cor)
plt.show()

df.isnull().sum()
# Pregnancies                 0
# Glucose                     0
# BloodPressure               0
# SkinThickness               0
# Insulin                     0
# BMI                         0
# DiabetesPedigreeFunction    0
# Age                         0
# Outcome                     0

def assing_missing_values(dataframe, except_cols):
    for col in dataframe.columns:
        dataframe[col] = [val if val!=0 or col in except_cols else np.nan for val in df[col].values]
    return dataframe

dff = assing_missing_values(df, except_cols=["Pregnancies", "Outcome"])

dff.isnull().any()
# Pregnancies                 False
# Glucose                      True
# BloodPressure                True
# SkinThickness                True
# Insulin                      True
# BMI                          True
# DiabetesPedigreeFunction    False
# Age                         False
# Outcome                     False

dff.isnull().sum()
# Pregnancies                   0
# Glucose                       5
# BloodPressure                35
# SkinThickness               227
# Insulin                     374
# BMI                          11
# DiabetesPedigreeFunction      0
# Age                           0
# Outcome                       0

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_cols = missing_values_table(dff, True)
#                n_miss    ratio
# Insulin           374 48.70000
# SkinThickness     227 29.56000
# BloodPressure      35  4.56000
# BMI                11  1.43000
# Glucose             5  0.65000
# ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

msno.bar(df)
plt.show()

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(dff, "Outcome", na_cols)
# Glucose_NA_FLAG
# 0                    0.34862    763
# 1                    0.40000      5
#                        TARGET_MEAN  Count
# BloodPressure_NA_FLAG
# 0                          0.34379    733
# 1                          0.45714     35
#                        TARGET_MEAN  Count
# SkinThickness_NA_FLAG
# 0                          0.33272    541
# 1                          0.38767    227
#                  TARGET_MEAN  Count
# Insulin_NA_FLAG
# 0                    0.32995    394
# 1                    0.36898    374
#              TARGET_MEAN  Count
# BMI_NA_FLAG
# 0                0.35139    757
# 1                0.18182     11

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
#    Outcome  Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin     BMI  DiabetesPedigreeFunction     Age
# 0  1.00000      0.35294  0.67097        0.48980        0.30435      NaN 0.31493                   0.23442 0.48333
# 1  0.00000      0.05882  0.26452        0.42857        0.23913      NaN 0.17178                   0.11657 0.16667
# 2  1.00000      0.47059  0.89677        0.40816            NaN      NaN 0.10429                   0.25363 0.18333
# 3  0.00000      0.05882  0.29032        0.42857        0.17391  0.09615 0.20245                   0.03800 0.00000
# 4  1.00000      0.00000  0.60000        0.16327        0.30435  0.18510 0.50920                   0.94364 0.20000

imputer = KNNImputer()
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
#    Outcome  Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin     BMI  DiabetesPedigreeFunction     Age
# 0  1.00000      0.35294  0.67097        0.48980        0.30435  0.38486 0.31493                   0.23442 0.48333
# 1  0.00000      0.05882  0.26452        0.42857        0.23913  0.05072 0.17178                   0.11657 0.16667
# 2  1.00000      0.47059  0.89677        0.40816        0.27391  0.26923 0.10429                   0.25363 0.18333
# 3  0.00000      0.05882  0.29032        0.42857        0.17391  0.09615 0.20245                   0.03800 0.00000
# 4  1.00000      0.00000  0.60000        0.16327        0.30435  0.18510 0.50920                   0.94364 0.20000

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()
#    Outcome  Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin      BMI  DiabetesPedigreeFunction      Age
# 0  1.00000      6.00000 148.00000       72.00000       35.00000 334.20000 33.60000                   0.62700 50.00000
# 1  0.00000      1.00000  85.00000       66.00000       29.00000  56.20000 26.60000                   0.35100 31.00000
# 2  1.00000      8.00000 183.00000       64.00000       32.20000 238.00000 23.30000                   0.67200 32.00000
# 3  0.00000      1.00000  89.00000       66.00000       23.00000  94.00000 28.10000                   0.16700 21.00000
# 4  1.00000      0.00000 137.00000       40.00000       35.00000 168.00000 43.10000                   2.28800 33.00000
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(dff, col)

dff.loc[df_["Pregnancies"] >= 10, "Pregnancies"] = 0.0
dff.loc[df_["Pregnancies"] == 0, "New_Preg_Cat"] = "Not Pregnant"
dff.loc[df_["Pregnancies"] > 0, "New_Preg_Cat"] = "Pregnant"

dff.loc[df_["Glucose"] < 70, "New_Glucose_Cat"] = "low"
dff.loc[((df_["Glucose"] < 100) & (df_["Glucose"] >= 70)), "New_Glucose_Cat"] = "Normal"
dff.loc[((df_["Glucose"] < 125) & (df_["Glucose"] >= 100)), "New_Glucose_Cat"] = "Potential"
dff.loc[df_["Glucose"] >= 125, "New_Glucose_Cat"] = "High"

dff.loc[df_["BloodPressure"] > 90, "New_Bloodpr_Cat"] = "High"
dff.loc[((df_["BloodPressure"] <= 90) & (df_["BloodPressure"] > 0)), "New_Bloodpr_Cat"] = "Normal"

dff.loc[df_["BMI"] < 18.5, "New_BMI_Cat"] = "Underweight"
dff.loc[((df_["BMI"] < 30) & (df_["BMI"] >= 18.5)), "New_BMI_Cat"] = "Normal"
dff.loc[((df_["BMI"] < 34.9) & (df_["BMI"] >= 30)), "New_BMI_Cat"] = "Obese"
dff.loc[df_["BMI"] >= 34.9, "New_BMI_Cat"] = "Extremely Obese"

dff.loc[df_["Age"] <= 21, "New_Age_Cat"] = "Young"
dff.loc[((df_["Age"] <= 50) & (df_["Age"] > 21)), "New_Age_Cat"] = "Mature"
dff.loc[df_["Age"] > 50, "New_Age_Cat"] = "Senior"

dff.head()
#    Outcome  Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin      BMI  DiabetesPedigreeFunction      Age  New_Preg_Cat New_Glucose_Cat New_Bloodpr_Cat      New_BMI_Cat New_Age_Cat
# 0  1.00000      6.00000 148.00000       72.00000       35.00000 334.20000 33.60000                   0.62700 50.00000      Pregnant            High          Normal            Obese      Mature
# 1  0.00000      1.00000  85.00000       66.00000       29.00000  56.20000 26.60000                   0.35100 31.00000      Pregnant          Normal          Normal           Normal      Mature
# 2  1.00000      8.00000 183.00000       64.00000       32.20000 238.00000 23.30000                   0.67200 32.00000      Pregnant            High          Normal           Normal      Mature
# 3  0.00000      1.00000  89.00000       66.00000       23.00000  94.00000 28.10000                   0.16700 21.00000      Pregnant          Normal          Normal           Normal       Young
# 4  1.00000      0.00000 137.00000       40.00000       35.00000 168.00000 43.10000                   1.20000 33.00000  Not Pregnant            High          Normal  Extremely Obese      Mature

dff.isnull().sum()
# Outcome                      0
# Pregnancies                  0
# Glucose                      0
# BloodPressure                0
# SkinThickness                0
# Insulin                      0
# BMI                          0
# DiabetesPedigreeFunction     0
# Age                          0
# New_Preg_Cat                 0
# New_Glucose_Cat              0
# New_Bloodpr_Cat             35
# New_BMI_Cat                  0
# New_Age_Cat                  0

cat_cols = [col for col in dff.columns if dff[col].dtypes == "O"]

def one_hot_encoder(dataframe, categorical_columns, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=drop_first)
    return dataframe

one_hot_encoder(dff, cat_cols)
#     Outcome  Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin      BMI  DiabetesPedigreeFunction      Age  New_Preg_Cat_Pregnant  New_Glucose_Cat_Normal  New_Glucose_Cat_Potential  New_Glucose_Cat_low  New_Bloodpr_Cat_Normal  New_BMI_Cat_Normal  New_BMI_Cat_Obese  New_BMI_Cat_Underweight  New_Age_Cat_Senior  New_Age_Cat_Young
# 0    1.00000      6.00000 148.00000       72.00000       35.00000 334.20000 33.60000                   0.62700 50.00000                      1                       0                          0                    0                       1                   0                  1                        0                   0                  0
# 1    0.00000      1.00000  85.00000       66.00000       29.00000  56.20000 26.60000                   0.35100 31.00000                      1                       1                          0                    0                       1                   1                  0                        0                   0                  0
# 2    1.00000      8.00000 183.00000       64.00000       32.20000 238.00000 23.30000                   0.67200 32.00000                      1                       0                          0                    0                       1                   1                  0                        0                   0                  0
# 3    0.00000      1.00000  89.00000       66.00000       23.00000  94.00000 28.10000                   0.16700 21.00000                      1                       1                          0                    0                       1                   1                  0                        0                   0                  1
# 4    1.00000      0.00000 137.00000       40.00000       35.00000 168.00000 43.10000                   1.20000 33.00000                      0                       0                          0                    0                       1                   0                  0                        0                   0                  0
# ..       ...          ...       ...            ...            ...       ...      ...                       ...      ...                    ...                     ...                        ...                  ...                     ...                 ...                ...                      ...                 ...                ...
# 763  0.00000      0.00000 101.00000       76.00000       48.00000 180.00000 32.90000                   0.17100 63.00000                      1                       0                          1                    0                       1                   0                  1                        0                   1                  0
# 764  0.00000      2.00000 122.00000       70.00000       27.00000 150.00000 36.80000                   0.34000 27.00000                      1                       0                          1                    0                       1                   0                  0                        0                   0                  0
# 765  0.00000      5.00000 121.00000       72.00000       23.00000 112.00000 26.20000                   0.24500 30.00000                      1                       0                          1                    0                       1                   1                  0                        0                   0                  0
# 766  1.00000      1.00000 126.00000       60.00000       26.40000 122.40000 30.10000                   0.34900 47.00000                      1                       0                          0                    0                       1                   0                  1                        0                   0                  0
# 767  0.00000      1.00000  93.00000       70.00000       31.00000  72.40000 30.40000                   0.31500 23.00000                      1                       1                          0                    0                       1                   0                  1                        0                   0                  0
# [768 rows x 19 columns]

dff = one_hot_encoder(dff, cat_cols)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

rare_analyser(dff, "Outcome", cat_cols)
# Outcome : 2
#          COUNT   RATIO  TARGET_MEAN
# 0.00000    500 0.65104      0.00000
# 1.00000    268 0.34896      1.00000
# New_Preg_Cat_Pregnant : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    111 0.14453      0.34234
# 1    657 0.85547      0.35008
# New_Glucose_Cat_Normal : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    587 0.76432      0.43271
# 1    181 0.23568      0.07735
# New_Glucose_Cat_Potential : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    508 0.66146      0.39173
# 1    260 0.33854      0.26538
# New_Glucose_Cat_low : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    752 0.97917      0.35372
# 1     16 0.02083      0.12500
# New_Bloodpr_Cat_Normal : 2
#    COUNT   RATIO  TARGET_MEAN
# 0     73 0.09505      0.46575
# 1    695 0.90495      0.33669
# New_BMI_Cat_Normal : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    487 0.63411      0.45380
# 1    281 0.36589      0.16726
# New_BMI_Cat_Obese : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    550 0.71615      0.30909
# 1    218 0.28385      0.44954
# New_BMI_Cat_Underweight : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    753 0.98047      0.35325
# 1     15 0.01953      0.13333
# New_Age_Cat_Senior : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    687 0.89453      0.33479
# 1     81 0.10547      0.46914
# New_Age_Cat_Young : 2
#    COUNT   RATIO  TARGET_MEAN
# 0    705 0.91797      0.37305
# 1     63 0.08203      0.07937

rare_encoder(dff, 0.01, cat_cols)
#      Outcome  Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin      BMI  DiabetesPedigreeFunction      Age  New_Preg_Cat_Pregnant  New_Glucose_Cat_Normal  New_Glucose_Cat_Potential  New_Glucose_Cat_low  New_Bloodpr_Cat_Normal  New_BMI_Cat_Normal  New_BMI_Cat_Obese  New_BMI_Cat_Underweight  New_Age_Cat_Senior  New_Age_Cat_Young
# 0    1.00000      6.00000 148.00000       72.00000       35.00000 334.20000 33.60000                   0.62700 50.00000                      1                       0                          0                    0                       1                   0                  1                        0                   0                  0
# 1    0.00000      1.00000  85.00000       66.00000       29.00000  56.20000 26.60000                   0.35100 31.00000                      1                       1                          0                    0                       1                   1                  0                        0                   0                  0
# 2    1.00000      8.00000 183.00000       64.00000       32.20000 238.00000 23.30000                   0.67200 32.00000                      1                       0                          0                    0                       1                   1                  0                        0                   0                  0
# 3    0.00000      1.00000  89.00000       66.00000       23.00000  94.00000 28.10000                   0.16700 21.00000                      1                       1                          0                    0                       1                   1                  0                        0                   0                  1
# 4    1.00000      0.00000 137.00000       40.00000       35.00000 168.00000 43.10000                   1.20000 33.00000                      0                       0                          0                    0                       1                   0                  0                        0                   0                  0
# ..       ...          ...       ...            ...            ...       ...      ...                       ...      ...                    ...                     ...                        ...                  ...                     ...                 ...                ...                      ...                 ...                ...
# 763  0.00000      0.00000 101.00000       76.00000       48.00000 180.00000 32.90000                   0.17100 63.00000                      1                       0                          1                    0                       1                   0                  1                        0                   1                  0
# 764  0.00000      2.00000 122.00000       70.00000       27.00000 150.00000 36.80000                   0.34000 27.00000                      1                       0                          1                    0                       1                   0                  0                        0                   0                  0
# 765  0.00000      5.00000 121.00000       72.00000       23.00000 112.00000 26.20000                   0.24500 30.00000                      1                       0                          1                    0                       1                   1                  0                        0                   0                  0
# 766  1.00000      1.00000 126.00000       60.00000       26.40000 122.40000 30.10000                   0.34900 47.00000                      1                       0                          0                    0                       1                   0                  1                        0                   0                  0
# 767  0.00000      1.00000  93.00000       70.00000       31.00000  72.40000 30.40000                   0.31500 23.00000                      1                       1                          0                    0                       1                   0                  1                        0                   0                  0
# [768 rows x 19 columns]

dff = rare_encoder(dff, 0.01, cat_cols)

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

dff = one_hot_encoder(dff, ohe_cols)

useless_cols = [col for col in dff.columns if dff[col].nunique() == 2 and (dff[col].value_counts() / len(dff) < 0.01).any(axis=None)]

rs = RobustScaler()

dff[num_cols] = rs.fit_transform(dff[num_cols])

dff.head()
#    Outcome  Glucose  BloodPressure  SkinThickness  Insulin      BMI  DiabetesPedigreeFunction      Age  New_Preg_Cat_Pregnant  New_Glucose_Cat_Normal  New_Glucose_Cat_Potential  New_Glucose_Cat_low  New_Bloodpr_Cat_Normal  New_BMI_Cat_Normal  New_BMI_Cat_Obese  New_BMI_Cat_Underweight  New_Age_Cat_Senior  New_Age_Cat_Young  Pregnancies_1.0  Pregnancies_2.0  Pregnancies_2.9999999999999996  Pregnancies_4.0  Pregnancies_5.0  Pregnancies_5.999999999999999  Pregnancies_7.0  Pregnancies_8.0  Pregnancies_9.0  Pregnancies
# 0  1.00000  0.75152        0.00000        0.37500 -0.23969  0.17204                   0.66536  1.23529                      1                       0                          0                    0                       1                   0                  1                        0                   0                  0                0                0                               0                0                0                              1                0                0                0      0.60000
# 1  0.00000 -0.77576       -0.33333        0.18750 -0.23969 -0.58065                  -0.05621  0.11765                      1                       1                          0                    0                       1                   1                  0                        0                   0                  0                1                0                               0                0                0                              0                0                0                0     -0.40000
# 2  1.00000  1.60000       -0.44444       -0.71875 -0.23969 -0.93548                   0.78301  0.17647                      1                       0                          0                    0                       1                   1                  0                        0                   0                  0                0                0                               0                0                0                              0                0                1                0      1.00000
# 3  0.00000 -0.67879       -0.33333        0.00000  0.49902 -0.41935                  -0.53725 -0.47059                      1                       1                          0                    0                       1                   1                  0                        0                   0                  1                1                0                               0                0                0                              0                0                0                0     -0.40000
# 4  1.00000  0.48485       -1.77778        0.37500  1.08055  1.19355                   5.00784  0.23529                      0                       0                          0                    0                       1                   0                  0                        0                   0                  0                0                0                               0                0                0                              0                0                0                0     -0.60000

y = dff["Outcome"]
X = dff.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=47)

rf_model = RandomForestClassifier(random_state=2).fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy_score(y_pred, y_test)
# 0.7619047619047619

print("Accuracy Score: " + f'{accuracy_score(y_pred, y_test):.2f}')
# Accuracy Score: 0.76

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
