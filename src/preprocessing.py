import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sleep_disorder = pd.read_csv("Sleep_Data_Sampled.csv")

sleep_disorder.head()
sleep_disorder.isnull().sum()
sleep_disorder.duplicated().sum()

sleep_disorder['Gender'].unique()
sleep_disorder['Occupation'].unique()
sleep_disorder['BMI Category'].unique()
sleep_disorder['Blood Pressure'].unique()
sleep_disorder['Sleep Disorder'].unique()
sleep_disorder['Sleep Disorder'].value_counts()

sleep_disorder.drop(columns=['Occupation'], inplace=True)
sleep_disorder.drop(columns=['Blood Pressure'],inplace=True)
sleep_disorder.drop(columns=['Daily Steps'],inplace=True)

sleep_disorder['Gender'] = sleep_disorder['Gender'].replace({'Male':1,'Female':0})
sleep_disorder['BMI Category'] = sleep_disorder['BMI Category'].replace({'Normal Weight':0,'Normal':1,'Overweight':2,'Obese':3})
sleep_disorder['Sleep Disorder'] = sleep_disorder['Sleep Disorder'].replace({'Healthy':0,'Sleep Apnea':1,'Insomnia':2})