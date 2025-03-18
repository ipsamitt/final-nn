import pytest
import numpy as np
from nn import nn # Import your NeuralNetwork class
from nn import preprocess # Import your NeuralNetwork class
from nn import io # Import your NeuralNetwork class


nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}, {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
sample_nn = nn.NeuralNetwork(nn_arch = nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="binary_cross_entropy")



# ======= SIMPLE AUTOENCODER ARCHITECTURE =======
autoencoder_arch = [
    {"input_dim": 3, "output_dim": 2, "activation": "relu"},  # Encoder
    {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"}  # Decoder
]

# ======= CREATE A SIMPLE NEURAL NETWORK INSTANCE =======
autoencoder = nn.NeuralNetwork(
    nn_arch=autoencoder_arch,
    lr=0.01,
    seed=42,
    batch_size=1,
    epochs=15,
    loss_function='binary_cross_entropy'
)


# ======= DEFINE TEST INPUT DATA =======
X_test = np.array([[0.1, 0.5, 0.9]])  # Single test sample
y_test = np.array([[0.2, 0.6, 0.8]])  # Target output


def test_single_forward():
    """Test single forward pass for a layer."""
    W = np.array([[0.2, 0.3], [0.4, 0.5]])
    b = np.array([[0.1], [0.2]])
    A_prev = np.array([[1, 2], [3, 4]])  # Input features
    
    A, Z = sample_nn._single_forward(W, b, A_prev, "relu")
    
    assert A.shape == (2, 2), "Output shape is incorrect"
    assert np.all(Z == np.dot(W, A_prev) + b), "Z computation is incorrect"


def test_forward():
    output, cache = autoencoder.forward(X_test)
    
    expected_output = np.array([[0.45528492, 0.48277804, 0.47316978]])
    
    # Use np.allclose() for floating-point comparisons
    assert np.allclose(output, expected_output, atol=1e-5), "Forward pass output does not match expected values!"

def test_single_backprop():
    """Test single backpropagation step."""
    W = np.array([[0.2, 0.3], [0.4, 0.5]])
    b = np.array([[0.1], [0.2]])
    Z = np.array([[0.5, -0.5], [1, -1]])
    A_prev = np.array([[1, 2], [3, 4]])
    dA_curr = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    dA_prev, dW, db = sample_nn._single_backprop(W, b, Z, A_prev, dA_curr, "sigmoid")

    assert dW.shape == W.shape, "dW shape is incorrect"
    assert db.shape == b.shape, "db shape is incorrect"
import numpy as np

def test_predict():
    y_pred = autoencoder.predict(X_test)
    
    expected_output = np.array([[0.45469429, 0.4831488,  0.47411744]])
    
    # Use np.allclose() to handle floating-point precision differences
    assert np.allclose(y_pred, expected_output, atol=0.01), "Prediction output does not match expected values!"


def test_binary_cross_entropy():
    """Test binary cross entropy loss function."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    loss = sample_nn._binary_cross_entropy(y, y_hat)
    
    expected_loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
    assert np.isclose(loss, expected_loss)

def test_binary_cross_entropy_backprop():
    """Test binary cross entropy backpropagation."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    dA = sample_nn._binary_cross_entropy_backprop(y, y_hat)

    expected_dA = -(y / (y_hat + 1e-9) - (1 - y) / (1 - y_hat + 1e-9))
    assert np.allclose(dA, expected_dA)

def test_mean_squared_error():
    """Test mean squared error loss function."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    loss = sample_nn._mean_squared_error(y, y_hat)

    expected_loss = np.mean(np.power(y - y_hat, 2))
    assert np.isclose(loss, expected_loss)

def test_mean_squared_error_backprop():
    """Test mean squared error backpropagation."""
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    dA = sample_nn._mean_squared_error_backprop(y, y_hat)

    expected_dA = 2 * (y_hat - y) / y.shape[0]
    assert np.allclose(dA, expected_dA)


def test_sample_seqs():
    """Placeholder for sample sequence testing."""
    pos_seqs = io.read_text_file("data/rap1-lieb-positives.txt")
    neg_seqs = io.read_fasta_file("data/yeast-upstream-1k-negative.fa")

    labels = ([True] * len(pos_seqs)) + [False] * (len(neg_seqs))

    all_seqs = pos_seqs + neg_seqs
    sampled_seqs, sampled_labels = preprocess.sample_seqs(all_seqs, labels)

    true_count = sum(bool(x) for x in sampled_labels)
    false_count = sum(not bool(x) for x in sampled_labels)
    assert(true_count == false_count)


def test_one_hot_encode_seqs():
    """Placeholder for one-hot encoding sequence testing."""
    test_seqs = ["AG", "CT"]
    encoded_seqs = [[1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0]]

    assert(encoded_seqs == preprocess.one_hot_encode_seqs(test_seqs))
