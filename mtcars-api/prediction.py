import pandas as pd
import numpy as np
import sklearn.linear_model as lm

mtcars = pd.read_csv('mtcars.csv')

col_imp = ["cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]
X = mtcars[col_imp]
y = mtcars["mpg"]

model = lm.LinearRegression()
model.fit(X,y)

def predict(dict_values, col_imp=col_imp, model=model):
    x = np.array([float(dict_values[col]) for col in col_imp])
    x = x.reshape(1, -1)
    y_pred = model.predict(x)[0]
    return y_pred
