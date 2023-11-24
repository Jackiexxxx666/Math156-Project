import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = 'a.us.txt'
df = pd.read_csv(file_path, delimiter=',')
date_and_close = df[['Date', 'Close']].values

train_size = int(len(date_and_close) * 0.8) 
train_data = date_and_close[:train_size]
validation_data = date_and_close[train_size:]

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
train_prices = scaler.fit_transform(train_data[:, 1].reshape(-1, 1))
validation_prices = scaler.transform(validation_data[:, 1].reshape(-1, 1))

# Create a function to generate time series data
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5  # Use data from the past 5 days to predict the next day
X_train, Y_train = create_dataset(train_prices, look_back)
X_validation, Y_validation = create_dataset(validation_prices, look_back)

# The format of reshaping the input data is [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

# Build a simple RNN model
model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(look_back, 1), activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs=100, batch_size=32)

predicted_validation = model.predict(X_validation)
predicted_validation = scaler.inverse_transform(predicted_validation)  # denormalization
real_validation = scaler.inverse_transform(Y_validation.reshape(-1, 1))  # denormalization

# Calculate performance metrics
mse = mean_squared_error(real_validation, predicted_validation)
print(f"MSE: {mse}")

# Plot predictions versus actuals
# plt.figure(figsize=(10, 6))
# plt.plot(real_validation, label='actual price', color='blue')
# plt.plot(predicted_validation, label='Predict price', color='red')
# plt.title('stock price prediction')
# plt.xlabel('time')
# plt.ylabel('close price')
# plt.legend()
# plt.show()

# 预测未来三个月的价格
num_predictions = 60  # 根据您的数据频率调整
predictions = []

# 获取最后一个训练数据窗口
last_window = train_prices[-look_back:]
current_batch = last_window.reshape((1, look_back, 1))

for i in range(num_predictions):
    # 使用当前批次预测下一个价格
    current_pred = model.predict(current_batch)[0]
    
    # 存储预测结果
    predictions.append(current_pred)
    
    # 为下一次预测更新批次
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# 反规范化预测
predicted_prices = scaler.inverse_transform(predictions)

# 生成未来日期的时间索引
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=num_predictions, freq='B')

# 可视化预测
plt.figure(figsize=(10, 6))
plt.plot(future_dates, predicted_prices, label='Predicted Future Price', color='red')
plt.title('Future Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()
