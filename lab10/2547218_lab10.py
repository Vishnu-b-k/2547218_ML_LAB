"""
Lab Exercise 10: Learning the XOR Boolean Function Using an MLP
Student Register No: 2547218
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(model, 'predict'):
        preds = model.predict(grid, verbose=0)
    else: 
        preds = model(tf.constant(grid, dtype=tf.float32)).numpy()
        
    Z = (preds >= 0.5).astype(int)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=100, cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')


def main():
    print("="*50)
    print("1. Dataset (XOR)")
    print("="*50)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    print("Input X:\\n", X)
    print("Output y:\\n", y)

    print("\\n"+"="*50)
    print("2. Keras Implementation (High-Level API)")
    print("="*50)
    keras_model = Sequential([
        Dense(8, input_dim=2, activation='relu', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    keras_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.05), metrics=['accuracy'])
    history = keras_model.fit(X, y, epochs=200, verbose=0)
    
    _, acc_keras = keras_model.evaluate(X, y, verbose=0)
    print(f"Keras Model Accuracy: {acc_keras * 100:.2f}%")
    preds_keras = keras_model.predict(X, verbose=0)
    preds_bin_keras = (preds_keras >= 0.5).astype(int)
    for i in range(len(X)):
        print(f"Input: {X[i]} | Actual: {y[i][0]} | Predicted: {preds_bin_keras[i][0]} (Prob: {preds_keras[i][0]:.4f})")

    print("\\n"+"="*50)
    print("3. Low-Level TensorFlow Implementation")
    print("="*50)
    tf.random.set_seed(42)
    W1 = tf.Variable(tf.random.normal([2, 8]), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([8]), dtype=tf.float32)
    W2 = tf.Variable(tf.random.normal([8, 1]), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

    def tf_model(X_input):
        hidden_out = tf.nn.relu(tf.matmul(X_input, W1) + b1)
        output = tf.nn.sigmoid(tf.matmul(hidden_out, W2) + b2)
        return output

    def bce_loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)

    optimizer_tf = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss_history_tf = []

    for epoch in range(200):
        with tf.GradientTape() as tape:
            y_pred = tf_model(X)
            loss = bce_loss(y, y_pred)
        gradients = tape.gradient(loss, [W1, b1, W2, b2])
        optimizer_tf.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
        loss_history_tf.append(loss.numpy())

    final_preds_tf = tf_model(X).numpy()
    preds_bin_tf = (final_preds_tf >= 0.5).astype(int)
    acc_tf = np.mean(preds_bin_tf == y)
    print(f"TF Low-Level Accuracy: {acc_tf * 100:.2f}%")
    for i in range(len(X)):
        print(f"Input: {X[i]} | Actual: {y[i][0]} | Predicted: {preds_bin_tf[i][0]} (Prob: {final_preds_tf[i][0]:.4f})")

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Keras Loss')
    plt.title('Keras Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plot_decision_boundary(keras_model, X, y, 'Keras Decision Boundary')

    plt.subplot(1, 3, 3)
    plot_decision_boundary(tf_model, X, y, 'TF Low-Level Boundary')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
