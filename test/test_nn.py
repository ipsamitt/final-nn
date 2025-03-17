import pytest
import numpy as np
from nn import nn # Import your NeuralNetwork class
from keras.losses import binary_crossentropy


nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}, {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
sample_nn = nn.NeuralNetwork(nn_arch = nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="binary_cross_entropy")

def test_single_forward():
    """Test single forward pass for a layer."""
    W = np.array([[0.2, 0.3], [0.4, 0.5]])
    b = np.array([[0.1], [0.2]])
    A_prev = np.array([[1, 2], [3, 4]])  # Input features
    
    A, Z = sample_nn._single_forward(W, b, A_prev, "relu")
    
    assert A.shape == (2, 2), "Output shape is incorrect"
    assert np.all(Z == np.dot(W, A_prev) + b), "Z computation is incorrect"

def test_forward():
    """Test full forward pass."""
    X = np.array([[0.5, 1.5]])
    y_hat, cache = sample_nn.forward(X)
    
    assert y_hat.shape == (1, 1), "Output shape is incorrect"

def test_single_backprop():
    """Test single backpropagation step."""
    W = np.array([[0.2, 0.3], [0.4, 0.5]])
    b = np.array([[0.1], [0.2]])
    Z = np.array([[0.5, -0.5], [1, -1]])
    A_prev = np.array([[1, 2], [3, 4]])
    dA_curr = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    dA_prev, dW, db = sample_nn._single_backprop(W, b, Z, A_prev, dA_curr, "relu")

    assert dW.shape == W.shape, "dW shape is incorrect"
    assert db.shape == b.shape, "db shape is incorrect"

def test_predict():
    """Test prediction function."""
    X = np.array([[0.5, 1.5]])
    y_hat = sample_nn.predict(X)
    
    assert y_hat.shape == (1, 1), "Prediction shape is incorrect"

def test_binary_cross_entropy():
    """Test binary cross entropy loss function."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    loss = sample_nn._binary_cross_entropy(y, y_hat)
    
    expected_loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
    assert np.isclose(loss, expected_loss), "Binary cross-entropy loss calculation is incorrect"

def test_binary_cross_entropy_backprop():
    """Test binary cross entropy backpropagation."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    dA = sample_nn._binary_cross_entropy_backprop(y, y_hat)

    expected_dA = -(y / (y_hat + 1e-9) - (1 - y) / (1 - y_hat + 1e-9))
    assert np.allclose(dA, expected_dA), "Binary cross-entropy backpropagation is incorrect"

def test_mean_squared_error():
    """Test mean squared error loss function."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    loss = sample_nn._mean_squared_error(y, y_hat)

    expected_loss = np.mean(np.power(y - y_hat, 2))
    assert np.isclose(loss, expected_loss), "MSE loss calculation is incorrect"

def test_mean_squared_error_backprop():
    """Test mean squared error backpropagation."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    dA = sample_nn._mean_squared_error_backprop(y, y_hat)

    expected_dA = 2 * (y_hat - y) / y.shape[0]
    assert np.allclose(dA, expected_dA), "MSE backpropagation is incorrect"

def test_sample_seqs():
    """Placeholder for sample sequence testing."""
    pass

def test_one_hot_encode_seqs():
    """Placeholder for one-hot encoding sequence testing."""
    pass
