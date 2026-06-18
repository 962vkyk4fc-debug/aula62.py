import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Simulação de Dados (Exemplo)
# Recursos: [Idade, Tempo de Contrato (meses), Gasto Mensal]
X = np.random.rand(1000, 3) * 100
y = np.random.randint(0, 2, size=(1000, 1))

# 2. Pré-processamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Construção do Modelo TensorFlow
model = tf.keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Saída binária (0 ou 1)
])

# 4. Compilação e Treino
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 5. Salvar o modelo para o Deploy
model.save('modelo_churn.h5') 