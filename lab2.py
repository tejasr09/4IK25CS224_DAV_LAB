import pandas as pd
import numpy as np
data = {"Name": ["Aarav", "Diya", "Arjun", "Ananya", "Rohan","Meera", "Vikram", "Ishita", "Karan", "Priya","Aditya", "Sneha", "Rahul", "Neha", "Aryan",
        "Kavya", "Sahil", "Pooja", "Varun", "Riya",
        "Nikhil", "Anjali", "Manish", "Simran", "Yash"],

    "Age": [21, 22, np.nan, 20, 23,21, 24, np.nan, 22, 20,25, 21, 23, np.nan, 22,24, 20, 21, 23, np.nan,22, 25, 24, 20, np.nan],

    "Marks": [85, 92, 78, np.nan, 88,95, 67, 81, np.nan, 90,73, 86, 79, 91, np.nan,84, 76, 89, 94, np.nan,82, 71, 87, 80, np.nan],

    "City": ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad","Pune", "Kolkata", "Bengaluru", "Delhi", "Mumbai","Chennai", "Pune", "Hyderabad", "Kolkata", "Delhi",
            "Bengaluru", "Mumbai", "Chennai", "Pune", "Hyderabad",
            "Kolkata", "Delhi", "Bengaluru", "Mumbai", "Pune"]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

print("\nMean:")
print(df[["Age", "Marks"]].mean())

print("\nMedian:")
print(df[["Age", "Marks"]].median())

print("\nStandard Deviation:")
print(df[["Age", "Marks"]].std())
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal number of missing values:")
print(df.isnull().sum().sum())

print("\nRows containing missing values:")
print(df[df.isnull().any(axis=1)])
print("\n" + "=" * 50)
print("DUPLICATE READINGS")
print("=" * 50)

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate rows:")
print(df[df.duplicated()])