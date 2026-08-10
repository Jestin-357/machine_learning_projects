import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## Data Collection and Preprocessing
### there is no header in the dataset so we have to set header = None

df = pd.read_csv('/Users/jestin/Desktop/machine_learning_projects/Rock_vs_Mine_prediction/Copy of sonar data.csv',header = None)

df.head()
df.info()
df.describe()   # describe the dataset

df.shape    # no. of rows and columns

df[60].value_counts()

# m - Mine
# r - Rock

df.groupby(60).mean()

# split the data into data and label
x = df.drop(columns=60,axis =1)
y = df[60]

print(x)
print(y)

# training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, stratify =y, random_state = 1) 

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print(x_train)
print(y_train)

# Model Training

model_1 = LogisticRegression()

model_1.fit(x_train, y_train)

# model Evaluation

### accuracy on training data
x_train_prediction = model_1.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

### accuracy on testing data

x_test_prediction = model_1.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy on testing data : ', testing_data_accuracy)
print('Accuracy on trainig data : ', training_data_accuracy)


# Making a predictive system

input_data = (0.1313,0.2339,0.3059,0.4264,0.4010,0.1791,0.1853,0.0055,0.1929,0.2231,0.2907,0.2259,0.3136,0.3302,0.3660,0.3956,0.4386,0.4670,0.5255,0.3735,0.2243,0.1973,0.4337,0.6532,0.5070,0.2796,0.4163,0.5950,0.5242,0.4178,0.3714,0.2375,0.0863,0.1437,0.2896,0.4577,0.3725,0.3372,0.3803,0.4181,0.3603,0.2711,0.1653,0.1951,0.2811,0.2246,0.1921,0.1500,0.0665,0.0193,0.0156,0.0362,0.0210,0.0154,0.0180,0.0013,0.0106,0.0127,0.0178,0.0231)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model_1.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
    print("the object is a Rock")
else:
    print("the object is a Mine")

    