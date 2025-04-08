import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

X = sleep_disorder[['Gender','Age','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','BMI Category','Heart Rate'
                    ]]
y = sleep_disorder['Sleep Disorder']

print("The X parameters are:\n",X)
print("The y parameters is:\n",y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=48,stratify=y)

scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

svm = SVC()
rf = RandomForestClassifier()

svm.fit(X_train_scaled,y_train)
rf.fit(X_train_scaled,y_train)