from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X = np.array([6,6,7,5,6,7,7]).reshape(-1,1)
y = np.array([3.25,3.25,3.25,3.25,3.25,3.25,3.25])


model = Sequential([
    Dense(2, activation='relu', input_shape=(1,)),
    Dense(1)])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'])

model.fit(X, y, epochs=100)

loss, mae = model.evaluate(X, y)
print("MAE:", mae)