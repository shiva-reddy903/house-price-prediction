import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = {
    "Area" : [1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200],
    "price" : [20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
    }
df = pd.DataFrame(data)
df["price"] = df["price"].fillna(df["price"].mean())
df.drop_duplicates(inplace=True)

x = df[["Area"]]
y = df["price"]

#splitting train,test data
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2
    )
#scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred_test = model.predict(x_test_scaled)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred_test)
print("r2_score :",score)

new_data = pd.DataFrame([[1700]],columns=["Area"])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("predicted price :" ,prediction)
y_pred_all = model.predict(x_train_scaled)
plt.scatter(x_train,y_train,color="blue",label="Actual data point")
plt.plot(x_train,y_pred_all,color="green",label="Regression line")
plt.xlabel("Area")
plt.ylabel("price")
plt.title("house price prediction")
plt.scatter(new_data,prediction,color="red",label="predicted point")
plt.legend()
plt.show()


 