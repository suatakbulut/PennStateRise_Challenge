# PennStateRise_Challenge

This repo aims to build a model to predict whether a patient will be hospitalized within 30 days of their index date. For privacy reasons, the data set is not presented here. Please make sure that the relative path of the data is `./data/dataset.csv` when running thie repo. 

## Data Wrangling and Feature Engineering

Using the patient’s medical history prior to their personal index date, a set of 26 features/covariates/predictors have been created. These features consist of:

- Binary values. 
example: `lab-test` -- (0 or 1)

- Categorical values. 
example: `birth_gender` -- ‘M’ or ‘F’

- Continuous values. 
example: `age` -- 61.7

- List of dates that the feature was detected in the patient’s EHR. 
  example: `smoking`  -- [20201017, 20201102], [0], "0"

  - `{col}_isPresent`: It takes the value 0, if there is no data, 1 if there is a value of 0, and 2 if there is a date data

- List of date and value pairs
  example: `aspirin` -- [[20210112.0, '15 MG'], [20210112.0, '15 MG]], [[2020112.0, 192.0], [20200514.0, "Unknown"]], [0], "0"

  - For each such col -- except `weight` and `height`,
    - `{col}_isPresent` is created similarly. 
    - `{col}_daily_total_value_avg`: average total daily dose
    - `{col}_daily_total_value_std`: std of total daily dose
    - `{col}_daily_total_value_trend_slope`: The slope of a simple linear fit on (total_daily_dose, date) plane

  - For weight and height (they are List of date and value pairs as well):
    - `bmi_avg`, `bmi_std`, `bmi_trend_slope`
    - `weight_avg`, `weight_std`, `weight_trend_slope`

For each observation  
  - `dates_nunique`: number of unique dates for which we have entry for the patient
  - `dates_range`: the range of these dates
  - `num_isPresents`: total number of columns for which the feature is present
  - `is_senior`: True if age > 75
  - `is_black`: True if race is Black or African American
  - and some interaction terms

## Training 
Multiple classification predictors are considered and trained using GridsearchCV method. 

![results_table](https://github.com/suatakbulut/PennStateRise_Challenge/assets/59936993/630e0a0a-1823-472b-bdfa-a06308e9951b)


To replicate the training simple run 
Run 
```
main.py -cd 1 -ts 0.4
```
on the terminal. To recreate the features set -cd to 1, and to adjust the train set ratio by changing -ts. The output will be saved in `trained_models` folder. 

## Post Analysis 
Please refer to the jupyter notebooks inside the `test_notebooks` folder. Feature Importance analysis conducted on the best-performing model, `Random Forest Classifier`, looks like as follows:

![feature_importance](https://github.com/suatakbulut/PennStateRise_Challenge/assets/59936993/8296afa7-8b55-4cc4-b01a-462eaa30d968)

Focusing attention on only those 10 features and fitting a Logistic Regression shows the significance of those features as well.

![logit_on_important_features](https://github.com/suatakbulut/PennStateRise_Challenge/assets/59936993/14c143fd-4267-4cc5-a3c2-657e5e1b0499)

Finally, re-trainin the best-performing model with the same parameters on these 10 features and comparing its results to the Logistic Regression's above is depicted below. 

![comparison_on_importantant_features](https://github.com/suatakbulut/PennStateRise_Challenge/assets/59936993/bfeebb62-9598-453d-81e6-9bfe46e22f50)



