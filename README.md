## NAME:RAMYA P
## REG NO:212223230168
## DATE:7-10-24


## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~
![image](https://github.com/user-attachments/assets/b09c2db8-a3f7-48f5-afec-84594af17387)

## ORDINAL ENCODER
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![image](https://github.com/user-attachments/assets/511ad6a6-7a92-4acf-a582-d6128f39016a)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![image](https://github.com/user-attachments/assets/17369483-60e1-42e7-b488-ea98c9131346)
## LABEL ENCODER
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
~~~
![image](https://github.com/user-attachments/assets/629e5fdb-45fb-47f0-8061-8e66e72e8a08)
~~~
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
![image](https://github.com/user-attachments/assets/6db5e2ea-95c5-4fe0-b0bd-452cbf18dc17)


## Onehot encoder
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
~~~
![image](https://github.com/user-attachments/assets/dbcb45c9-abf7-41f1-addb-ab9f6586349c)
~~~
df2=pd.concat([df,enc],axis=1)
df2
~~~
![image](https://github.com/user-attachments/assets/f0c290ec-59e1-4a9d-bf92-712ec7f6808a)
~~~
pip install --upgrade category_encoders
~~~
![image](https://github.com/user-attachments/assets/17605af3-62c2-4738-96c2-db7cfd713365)

## BinaryEncoder
~~~
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df
~~~
![image](https://github.com/user-attachments/assets/5ba3afec-f9ca-471f-997c-2c23a88172d6)
~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
~~~
![image](https://github.com/user-attachments/assets/dbbfb7a7-719e-4193-aaf8-485dcf02d96a)
## Target encoder
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
~~~
![image](https://github.com/user-attachments/assets/f46a5506-59c3-4877-846d-e923b85cb31d)


## Feature engineering
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
~~~
![image](https://github.com/user-attachments/assets/24efa980-30b1-42fb-9f73-8da50dfef9f3)
~~~
df.skew()
![image](https://github.com/user-attachments/assets/f41b15b4-a970-413f-9f4d-fec37fda8f54)
~~~

df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/df60af58-d4e0-4779-8a90-22719315d588)

~~~

df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/e879686e-29b6-4e76-bb5b-59d791e6c666)

~~~

df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/c792bdcf-8ceb-4e94-9549-ba3414a8b533)

~~~


df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/58e66009-ddc0-480e-9fec-fa1ed23daedd)
~~~


## Power transformation
~~~

df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/b1df3e56-9ae3-4aca-9a18-a44d4a77c875)


~~~

df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
~~~

~~~

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/60d2f715-d462-4844-8fe1-15c5bb00562a)

~~~

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/9b99332b-ccca-410b-9215-845473e66227)

~~~

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/db91b2f0-8fd3-4f21-8e61-0a0c6348186e)





























   
## RESULT:

   Thus,the given data are read and Feature Encoding and Transformation process are performed and the data is saved to the file.




       
