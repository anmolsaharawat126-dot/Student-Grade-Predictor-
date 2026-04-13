import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'area': [1000, 1200, 1500, 1800, 2000],
    'bedrooms': [2, 3, 3, 4, 4],
    'bathrooms': [1, 2, 2, 3, 3],
    'price': [5000000, 6500000, 7200000, 9000000, 10000000]
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print("House Price Predictor")

a = float(input("Enter area: "))
b = int(input("Enter bedrooms: "))
c = int(input("Enter bathrooms: "))

p = model.predict([[a, b, c]])


print("Price is:", int(p[0]))