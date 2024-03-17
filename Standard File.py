#!/usr/bin/env python
# coding: utf-8

# Step 1: Prepare Data

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('C:\\Users\\nhart\\OneDrive\\Desktop\\IBM\\AI Engineer\\Introduction to Deep Learning with Keras\\Final Assignment\\concrete_data.csv')  # Update this path to your dataset's actual location

# Assuming the target variable is the last column in the dataframe
X = df.iloc[:, :-1].values  # Features: all columns except the last one
y = df.iloc[:, -1].values  # Target: the last column

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 2: Define Model

# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(n_hidden_layers=1, layer_size=32, input_shape=(8,)):  # Update this based on the actual number of features
    model = Sequential()
    model.add(Input(shape=input_shape))
    for _ in range(n_hidden_layers):
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Assuming it's a regression model
    return model


# Step 3: Compile Model

# In[21]:


model = build_model(n_hidden_layers=1, input_shape=(X_train_scaled.shape[1],))
model.compile(optimizer='adam', loss='mean_squared_error')


# Step 4: Train Model

# In[22]:


history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, verbose=0)


# Step 5: Evaluate Model

# In[23]:


test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Loss: {test_loss}')

