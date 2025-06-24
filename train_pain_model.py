import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# تحميل بيانات التدريب
df = pd.read_csv("plot_values_with_pain_level.csv")
df.dropna(inplace=True)

# ميزات التدريب
X = df[['ECG_Avg', 'Flex_Avg']].values
y = df['Pain_Level'].values

# تحويل القيم
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# تحويل التصنيف إلى one-hot
y_categorical = to_categorical(y, num_classes=3)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# بناء النموذج
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=1),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# حفظ النموذج
model.save("pain_prediction_cnn_model.h5")

# حفظ السكيلر
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("✅ النموذج والسكيلر تم حفظهم بنجاح.")
