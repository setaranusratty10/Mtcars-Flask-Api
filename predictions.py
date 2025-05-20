import pandas as pd
import numpy as np
import sklearn.linear_model as lm

mtcars = pd.read_csv('/Users/setaranusratty/Downloads/mtcars.csv')
mtcars.head()

my_model = lm.LinearRegression()
X = mtcars[["cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]]
y = mtcars["mpg"]
my_model.fit(X,y)

y_hat = my_model.predict(X)
print(f"The RMSE of the model is {np.sqrt(np.mean((y-y_hat)**2))}")
