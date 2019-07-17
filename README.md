# Risk Profile Predictor

## Goal

The goal of this project is to build a multiclass classifier that determines people's risk profile (from 1 to 8) for an insurance company based on a number of features.

## The data

This project uses a (modified) dataset from Kaggle. The dataset has been thoroughly anonymised, which makes it extra challenging.

**Variable - Description**
- Id - A unique identifier associated with an application
- Product_Info_1-7 - A set of normalized variables relating to the product applied for
- Ins_Age - Normalized age of applicant
- Ht - Normalized height of applicant
- Wt - Normalized weight of applicant
- BMI - Normalized BMI of applicant
- Employment_Info_1-6 - A set of normalized variables relating to the employment history of the applicant.
- InsuredInfo_1-6 - A set of normalized variables providing information about the applicant.
- Insurance_History_1-9 - A set of normalized variables relating to the insurance history of the applicant.
- Family_Hist_1-5 - A set of normalized variables relating to the family history of the applicant.
- Medical_History_1-41 - A set of normalized variables relating to the medical history of the applicant.
- Medical_Keyword_1-48 - A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.
- Response - This is the target variable, an ordinal variable relating to the final decision associated with an application

**The following variables are all categorical (nominal):**
```
Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41
```

**The following variables are continuous:**
```
Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5
```

**The following variables are discrete:**
```
Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
Medical_Keyword_1-48 are dummy variables.
```

## The model

The Risk Profile Predictor uses a Random Forest Classifier to assign a risk category to each client. Appropriate handling of null values for each variable is a key element of this model.

## Structure

model.py: this file contains the InsuranceModel class with the following main methods:
- preprocess_training_data: preprocessing steps for training data
- Preprocess_unseen_data: preprocessing steps for testing data
- fit: method to train the model
- predict: method to generate predictions


run.py: python script to download the data, train the model and test it. Once trained, the model is serialised in a pickle file.

## Setup

Download the python files of the project and execute ‘python run.py setup’ to download the source data into the ‘data’ directory.

## Usage

python run.py train : Instanciates and trains the Risk Profile Predictor Model.

python run.py test : Generates predictions for the test data using a trained Risk Profile Predictor Model
