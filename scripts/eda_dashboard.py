import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'C:/Users/dell/AlphaCare-Insurance-Solutions/notebooks/MachineLearningRating_v3.txt'  # Replace with the actual path to your file
data = pd.read_csv(file_path, delimiter='|')  # Adjust the delimiter if needed

# Title of the app
st.title('Insurance Data EDA')

# Display first few rows of the dataset
st.subheader('Data Preview')
st.dataframe(data.head())

# Data type summary
st.subheader('Data Types')
st.write(data.dtypes)

# Descriptive statistics
st.subheader('Descriptive Statistics')
st.write(data[['TotalPremium', 'TotalClaims']].describe())

# Univariate analysis (histograms)
st.subheader('Histograms')
fig, ax = plt.subplots()
sns.histplot(data['TotalPremium'], kde=True, ax=ax)
ax.set_title('Distribution of TotalPremium')
st.pyplot(fig)

fig, ax = plt.subplots()
sns.histplot(data['TotalClaims'], kde=True, ax=ax)
ax.set_title('Distribution of TotalClaims')
st.pyplot(fig)

# Bivariate analysis (scatter plot)
st.subheader('Scatter Plot: TotalPremium vs TotalClaims')
fig, ax = plt.subplots()
sns.scatterplot(x=data['TotalPremium'], y=data['TotalClaims'], hue=data['Province'], ax=ax)
ax.set_title('TotalPremium vs TotalClaims')
st.pyplot(fig)

# Correlation matrix
st.subheader('Correlation Matrix')
fig, ax = plt.subplots()
correlation_matrix = data[['TotalPremium', 'TotalClaims']].corr()
sns.heatmap(correlation_matrix, annot=True, ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)

# Outlier detection (box plot)
st.subheader('Box Plots')
fig, ax = plt.subplots()
sns.boxplot(x=data['TotalPremium'], ax=ax)
ax.set_title('Box plot of TotalPremium')
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x=data['TotalClaims'], ax=ax)
ax.set_title('Box plot of TotalClaims')
st.pyplot(fig)
