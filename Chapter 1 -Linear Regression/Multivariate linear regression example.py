import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'a': [2600, 3000, 3200, 3600, 4000],
    'b': [3, 4, 1, 3, 5],
    'c': [20, 15, 18, 30, 8],
    'd': [550000, 565000, 610000, 595000, 760000]
}
df = pd.DataFrame(data)
print(df)

# ==============================

from sklearn import linear_model
model = linear_model.LinearRegression()

# training
model.fit(df[['a', 'b', 'c']], df['d'])

# slope m1, m2 & m3
print(model.coef_)

# intercept b
print(model.intercept_)

# ==============================

# prediction a: 3200, b: 2, c: 10
print(model.predict([[3200, 2, 10]]))

# ===========================

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure('Multivariate Regression')
ax = plt.subplot(111, projection = '3d')

# plot dataset
plot = ax.scatter(
    df['a'],
    df['b'],
    df[['c']],
    c = df['d'],
    marker = 'o',
    s = 150,
    cmap = 'hot'
)

# plot prediction
ax.scatter(
    df['a'],
    df['b'],
    model.predict(df[['a', 'b', 'c']]),
    color = 'green',
    marker = '*',
    s = 150
)

fig.colorbar(plot)
ax.set_xlabel('aa')
ax.set_ylabel('bb')
ax.set_zlabel('cc')

plt.title('Multivariate Regression')
plt.show()
