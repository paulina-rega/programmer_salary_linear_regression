import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('salaries.csv', sep=";")

df = df.dropna(axis="rows")
df['years_of_experience'] = (df['years_of_experience'].replace(',','.', regex=True).astype(float))

salary = df['salary']
experience = df['years_of_experience']

X = np.array(experience).reshape(-1,1)
Y = np.array(salary)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

p = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Training dataset")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('test dataset')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

a=(regressor.coef_[0])
b=(regressor.intercept_)
