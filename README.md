### Feature Engineering

#### Bussiness Problem

##### It is desired to develop a machine learning model that can predict whether people
##### have diabetes when their characteristics are specified. You are expected to perform
##### the necessary data analysis and feature engineering steps before developing the model.

#### Dataset Story

##### The dataset is part of the large dataset held at the National Institutes of
##### Diabetes-Digestive-Kidney Diseases in the USA. Data used for diabetes research
##### on Pima Indian women aged 21 and over living in Phoenix, the 5th largest city
##### of the State of Arizona in the USA.
##### The target variable is specified as "outcome"; 1 indicates positive diabetes
##### test result, 0 indicates negative.

##### Pregnancies : Number of pregnancies
##### Glucose : 2-hour plasma glucose concentration in the oral glucose tolerance test
##### Blood Pressure : Blood Pressure (Low blood pressure) (mmHg)
##### SkinThickness : Skin Thickness
##### Insulin : 2-hour serum insulin (mu U/ml)
##### DiabetesPedigreeFunction : Function (2 hour plasma glucose concentration in oral glucose tolerance test)
##### BMI : Body Mass Index
##### Age : Age(year)
##### Outcome : Have the disease (1) or not (0)

#### Project Tasks

##### Step 1 : Examine the overall picture.
##### Step 2 : Capture the numeric and categorical variables.
##### Step 3 :  Analyze the numerical and categorical variables.
##### Step 4 : Perform target variable analysis. (The mean of the target variable according
##### to the categorical variables, the mean of the numeric variables according to the target variable)
##### Step 5 : Analyze the outlier observation.
##### Step 6 : Perform missing observation analysis.
##### Step 7 : Perform correlation analysis.

#### Task 2 : Feature Engineering

##### Step 1 : Take necessary actions for missing and outlier values. There are no missing observations in the data set,
##### but Glucose, Insulin etc. Observation units containing 0 in the variables may represent the missing value.
##### E.g; a person's glucose or insulin value will not be 0. Considering this situation, you can assign the zero values
##### to the relevant values as NaN and then apply the operations to the missing values.
##### Step 2 : Create new variables.
##### Step 3 : Perform the encoding operations.
##### Step 4 : Standardize for numeric variables.
##### Step 5 : Create the model.
