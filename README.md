#  Ex.No-6-Feature-Transformation
# AIM:
  To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
  Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
## STEP-1:
 Read the given Data.
## STEP-2:
  Clean the Data Set using Data Cleaning Process.
## STEP-3:
  Apply Feature Transformation techniques to all the feature of the data set.
## STEP-4:
Save the data to the file.

# CODE:
## FUNCTION TRANSFORMATION:
## LOG TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df['ModeratePositiveSkew']=np.log(df.ModeratePositiveSkew)

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')

plt.show()

## RECIPROCAL TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

df['HighlyPositiveSkew']=1/df.HighlyPositiveSkew

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df['HighlyNegativeSkew']=1/df.HighlyNegativeSkew
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew']=1/df.ModeratePositiveSkew
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew']=1/df.ModerateNegativeSkew
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

## SQUARE ROOT TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

df['HighlyPositiveSkew']=np.sqrt(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df['ModeratePositiveSkew']=np.sqrt(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

## POWER TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import PowerTransformer

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

transformer=PowerTransformer("yeo-johnson")

df['HighlyPositiveSkew']=pd.DataFrame(transformer.fit_transform(df[['HighlyPositiveSkew']]))

sm.qqplot(df['HighlyPositiveSkew'],line='45')

plt.show()

df['HighlyNegativeSkew']=pd.DataFrame(transformer.fit_transform(df[['HighlyNegativeSkew']]))

sm.qqplot(df['HighlyNegativeSkew'],line='45')

plt.show()

df['ModeratePositiveSkew']=pd.DataFrame(transformer.fit_transform(df[['ModeratePositiveSkew']]))

sm.qqplot(df['ModeratePositiveSkew'],line='45')

plt.show()

df['ModerateNegativeSkew']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df['ModerateNegativeSkew'],line='45')

plt.show()

## QUANTILE TRANSFORMATION:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

from google.colab import files

uploaded = files.upload()

df=pd.read_csv("Data_to_Transform1.csv")

qt=QuantileTransformer(output_distribution='normal')

df['HighlyPositiveSkew']=pd.DataFrame(qt.fit_transform(df[['HighlyPositiveSkew']]))

sm.qqplot(df['HighlyPositiveSkew'],line='45')

plt.show()

df['HighlyNegativeSkew']=pd.DataFrame(qt.fit_transform(df[['HighlyNegativeSkew']]))

sm.qqplot(df['HighlyNegativeSkew'],line='45')

plt.show()

df['ModeratePositiveSkew']=pd.DataFrame(qt.fit_transform(df[['ModeratePositiveSkew']]))

sm.qqplot(df['ModeratePositiveSkew'],line='45')

plt.show()

df['ModerateNegativeSkew']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df['ModerateNegativeSkew'],line='45')

plt.show()

# OUTPUT:
## FUNCTION TRANSFORMATION:
## LOG TRANSFORMATION:

![Screenshot (34)](https://user-images.githubusercontent.com/128498431/233117119-0b079530-1bdd-488d-9f99-11bcaf45e18b.png)

![Screenshot (35)](https://user-images.githubusercontent.com/128498431/233117234-2a0d1114-5e89-41bd-92cc-554198a0aa7e.png)

## RECIPROCAL TRANSFORMATION:
![Screenshot (36)](https://user-images.githubusercontent.com/128498431/233117299-7963013a-7669-45c9-8580-8e37873082e6.png)

![Screenshot (37)](https://user-images.githubusercontent.com/128498431/233117341-86862f7f-9b9e-4809-bba6-4e8b447001f9.png)

![Screenshot (38)](https://user-images.githubusercontent.com/128498431/233117386-f111a3aa-516f-4978-bd10-8f096bb5b3eb.png)

![Screenshot (38)](https://user-images.githubusercontent.com/128498431/233117444-ad864532-217a-4c4f-a484-f80bc5e729b6.png)

## SQUARE ROOT TRANSFORMATION:
![Screenshot (40)](https://user-images.githubusercontent.com/128498431/233117510-8cb7d5d0-1c44-497f-a920-d0cb32da6fca.png)

![Screenshot (40)](https://user-images.githubusercontent.com/128498431/233117564-a862e998-eb0c-4455-82ce-b332ee792e50.png)

## POWER TRANSFORMATION:
![Screenshot (42)](https://user-images.githubusercontent.com/128498431/233117607-d7b485db-851b-4faa-b859-c097eadb70e0.png)

![Screenshot (44)](https://user-images.githubusercontent.com/128498431/233117723-b389f33c-f3c2-4b15-8750-5615ec1cab80.png)

![Screenshot (45)](https://user-images.githubusercontent.com/128498431/233117790-e2e0be77-d1eb-4ab7-a6b6-bbcc028efec4.png)

![Screenshot (46)](https://user-images.githubusercontent.com/128498431/233117845-2e3dbd4c-999e-4f13-9394-ddef3b6d9436.png)

## QUANTILE TRANSFORMATION:
![Screenshot (43)](https://user-images.githubusercontent.com/128498431/233117680-4c6e8554-8616-4489-8fc1-ce337f1c7f69.png)

![Screenshot (47)](https://user-images.githubusercontent.com/128498431/233117900-ce01bb00-80ea-4f92-be2f-8fee3a3f1c82.png)

![Screenshot (48)](https://user-images.githubusercontent.com/128498431/233117950-ddb95584-147c-470c-957f-91b51e278ee7.png)

![Screenshot (49)](https://user-images.githubusercontent.com/128498431/233118009-cb226277-74b8-4223-a58c-7b1751cec76a.png)

# RESULT:
  Thus, the Feature Transformation for the given data set is executed and output was verified successfully.
