import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Iris Dataset Analysis", layout="wide")

# Title
st.title("Iris Dataset Analysis")

# Load the dataset
df = pd.read_csv("Iris.csv")

# Sidebar for dataset information
st.sidebar.header("Dataset Information")
st.sidebar.write(f"**Number of rows:** {df.shape[0]}")
st.sidebar.write(f"**Number of columns:** {df.shape[1]}")

# Show dataset
if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Iris Dataset")
    st.write(df)

# Sidebar for summary statistics
st.sidebar.header("Summary Statistics")
if st.sidebar.checkbox("Show summary statistics", False):
    st.subheader("Summary Statistics")
    st.write(df.describe())

# Sidebar for missing values
st.sidebar.header("Missing Values")
if st.sidebar.checkbox("Show missing values", False):
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Visualizations
st.header("Visualizations")

# Species count
st.subheader("Count of Species")
fig, ax = plt.subplots()
sns.countplot(x='Species', data=df, ax=ax)
st.pyplot(fig)

# Scatter plots
st.subheader("Scatter Plots")

# Scatter plot SepalLengthCm vs SepalWidthCm
st.write("### Sepal Length vs Sepal Width")
fig, ax = plt.subplots()
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df, ax=ax)
st.pyplot(fig)

# Scatter plot PetalLengthCm vs PetalWidthCm
st.write("### Petal Length vs Petal Width")
fig, ax = plt.subplots()
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df, ax=ax)
st.pyplot(fig)

# Pairplot
st.subheader("Pairplot")
fig = sns.pairplot(df.drop(['Id'], axis=1), hue='Species', height=2)
st.pyplot(fig)

# Histograms
st.subheader("Histograms")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].set_title("Sepal Length")
axes[0, 0].hist(df['SepalLengthCm'], bins=7)

axes[0, 1].set_title("Sepal Width")
axes[0, 1].hist(df['SepalWidthCm'], bins=5)

axes[1, 0].set_title("Petal Length")
axes[1, 0].hist(df['PetalLengthCm'], bins=6)

axes[1, 1].set_title("Petal Width")
axes[1, 1].hist(df['PetalWidthCm'], bins=6)

st.pyplot(fig)

# Distribution plots
st.subheader("Distribution Plots")
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
for feature in features:
    st.write(f"### {feature}")
    fig = sns.FacetGrid(df, hue="Species").map(sns.histplot, feature, kde=True).add_legend()
    st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
numerical_features = df.drop(['Id', 'Species'], axis=1)
fig, ax = plt.subplots()
sns.heatmap(numerical_features.corr(), annot=True, ax=ax)
st.pyplot(fig)

# Box plots
st.subheader("Box Plots")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

sns.boxplot(x="Species", y="SepalLengthCm", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Sepal Length")

sns.boxplot(x="Species", y="SepalWidthCm", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Sepal Width")

sns.boxplot(x="Species", y="PetalLengthCm", data=df, ax=axes[1, 0])
axes[1, 0].set_title("Petal Length")

sns.boxplot(x="Species", y="PetalWidthCm", data=df, ax=axes[1, 1])
axes[1, 1].set_title("Petal Width")

st.pyplot(fig)

# Handling outliers for SepalWidthCm
st.subheader("Handling Outliers: Sepal Width")
Q1 = np.percentile(df['SepalWidthCm'], 25, interpolation='midpoint')
Q3 = np.percentile(df['SepalWidthCm'], 75, interpolation='midpoint')
IQR = Q3 - Q1

upper = np.where(df['SepalWidthCm'] >= (Q3 + 1.5 * IQR))
lower = np.where(df['SepalWidthCm'] <= (Q1 - 1.5 * IQR))

# Display old shape
st.write(f"Old Shape: {df.shape}")

# Removing outliers
df_cleaned = df.copy()
df_cleaned.drop(upper[0], inplace=True)
df_cleaned.drop(lower[0], inplace=True)

# Display new shape
st.write(f"New Shape: {df_cleaned.shape}")

# Box plot after removing outliers
fig, ax = plt.subplots()
sns.boxplot(x='SepalWidthCm', data=df_cleaned, ax=ax)
st.pyplot(fig)
