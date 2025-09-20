import pandas as pd 
import numpy as np 
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt 

df = pd.read_csv(r'C:/Users/Mahbir Ahmed Maheen/Desktop/Materials Project/Mechanical_Properties.csv')
'''print(df.head(10))
print(df.info())
print(df.describe())
print(df.columns)'''

X = df[['Std','ID','Material','Heat treatment','Su', 'Sy', 'A5', 'Bhn', 'E', 'G','mu','Ro','pH', 'Desc', 'HV']]
Y = df['Prediction']

# Selecting the numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculating  correlation matrix
corr = numeric_df.corr()

# --- Seaborn Heatmap Visualization ---
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, linecolor='black',
            cbar_kws={'label': 'Correlation Coefficient', 'orientation': 'vertical'},
            square=True, ax=ax)

# title
fig.suptitle("Correlation Among Material Properties", fontsize=16, fontweight='bold')

# Rotating for better readability 
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)


plt.subplots_adjust(top=0.88)  # Adjust top margin to avoid clipping the title

plt.show()

# --- Pandas Styled Table with Conditional Formatting ---

# Defining function to highlight text color where correlation magnitude > 0.7
def highlight(val):
    color = 'red' if abs(val) > 0.7 else 'black'
    return f'color: {color}'

# Applying the style to the correlation matrix dataframe
styled_corr = corr.style.applymap(highlight)

# Displaying styled table (in Jupyter or similar environments)
display(styled_corr)

# Saving styled table to an HTML file
# styled_corr.to_html("correlation_matrix_styled.html")

