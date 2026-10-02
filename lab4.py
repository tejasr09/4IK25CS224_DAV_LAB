from pathlib import Path

import pandas as pd

data_path = Path(__file__).with_name("student_performance_dataset.csv")
df = pd.read_csv(data_path)
print("First 5 rows of dataset:")
print(df.head())
print("\nMissing values before handling:")
print(df.isnull().sum())
df.loc[0:5, 'Study_Hours_per_Week'] = None
df.loc[10:15, "Attendance_Rate"] = None
df.loc[20:25, "Past_Exam_Scores"] = None
df.loc[30:35, "Final_Exam_Score"] = None
print("\nMissing values after inserting sample Nans:")
print(df.isnull().sum()[df.isnull().sum() > 0])
df['Study_Hours_per_Week'] = df['Study_Hours_per_Week'].fillna(df['Study_Hours_per_Week'].mean())
df['Attendance_Rate'] = df['Attendance_Rate'].fillna(df['Attendance_Rate'].median())
df['Past_Exam_Scores'] = df['Past_Exam_Scores'].ffill()
df['Final_Exam_Score'] = df['Final_Exam_Score'].fillna(df['Final_Exam_Score'].mean())
print("\nMissing values after imputation:")
print(df.isnull().sum())
df = pd.get_dummies(df, columns=['Parental_Education_Level'])
print("\nOne-Hot Encoded Data:")
print(df.columns.tolist())
df["Pass_Fail_Encoded"] = df["Pass_Fail"].astype("category").cat.codes
print("\nLabel Encoded Data:")
print(df[['Pass_Fail', 'Pass_Fail_Encoded']].head())
df["Gender_Encoded"] = df["Gender"].astype("category").cat.codes
df["Internet_Access_Encoded"] = (
    df["Internet_Access_at_Home"].astype("category").cat.codes
)
df["Extracurricular_Encoded"] = (
    df["Extracurricular_Activities"].astype("category").cat.codes
)
output_path = Path(__file__).with_name("students_performance_cleaned.csv")
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved as '{output_path.name}'")