import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/student_performance_dataset.csv")
print("First 5 rows of Dataset:")
print(df.head())
print("Statistical Summary:")
print(df.describe())
num=["Study_Hours_per_Week","Attendance_Rate","Past_Exam_Scores","Final_Exam_Score"]
df.loc[0,"Study_Hours_per_Week"]=120
df.loc[1,"Attendance_Rate"]=250
df.loc[2,"Past_Exam_Scores"]=-20
df.loc[3,"Final_Exam_Score"]=200
print("Inserted a few artificial outlier values for demonstration")
print("IQR METHOD")
for col in num:
  q1=df[col].quantile(0.25)
  q2=df[col].quantile(0.75)
  iqr=q2-q1
  l=q1-1.5*iqr
  u=q2+1.5*iqr
  outlier=df[(df[col]<l)|(df[col]>u)]
  print(f"{col}: {len(outlier)} outlier(s) found (limits: {l:.2f} to {u:.2f})")
print("Z score")
for col in num:
  z_scores=(df[col]-df[col].mean())/df[col].std()
  outliers=df[z_scores.abs()>3]
  print(f"{col}: {len(outliers)} outlier(s) found (|z|>3)")
df_treated=df.copy()
print("Treating outilers")
for col in num:
  q1=df[col].quantile(0.25)
  q3=df[col].quantile(0.75)
  iqr=q3-q1
  df_treated[col]=df_treated[col].clip(lower=q1-1.5*iqr,upper=q3+1.5*iqr)
print("After treating outliers:")
print(df_treated.describe())
print("Statistics Before Treatment:")
print(df[num].describe())
print("Statistics After Treatment:")
print(df_treated[num].describe())
df_treated.to_csv("student_performance_dataset_treated.csv",index=False)
print("Data saved to student_performance_dataset_treated.csv")
print("Data saved successfully")
