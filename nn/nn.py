# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        
        #current layer linear transformed matrix
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        #relu activation
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        #sigmoid activation
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        #current input
        A_curr = X.T  # Transpose to match weight dimensions
        cache = {}
        #for every layer in network
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']

            #find activation method and apply, change current layer and output linear layer transformed matrix
            activation = layer['activation']
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, activation)
            cache[f'A{layer_idx-1}'] = A_curr
            cache[f'Z{layer_idx}'] = Z_curr
        return A_curr.T, cache 
       
    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        

        #relu activation
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        #sigmoid activation
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
       
        if dZ_curr.shape[1] != A_prev.shape[0]:
            dW_curr = np.dot(dZ_curr.T, A_prev) 
        else:
            dW_curr = np.dot(dZ_curr, A_prev) 

        db_curr = np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.dot(dZ_curr, W_curr)
        
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        grad_dict = {}
        if self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)

        elif self._loss_func == 'mean_squared_error':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        for i in reversed(range(1, len(self.arch) + 1)):
            W_curr = self._param_dict['W' + str(i)]
            b_curr = self._param_dict['b' + str(i)]
            Z_curr = cache['Z' + str(i)]
            A_prev = cache['A' + str(i - 1)] if i - 1 > 0 else y_hat 

            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, self.arch[i - 1]['activation'])

            grad_dict['dW' + str(i)] = dW_curr
            grad_dict['db' + str(i)] = db_curr

            dA_curr = dA_prev
        return grad_dict

    
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        #over each index, update parameters
        for idx in range(len(self.arch)):
            layer_idx = idx + 1
            self._param_dict[f'W{layer_idx}'] -= self._lr * grad_dict[f'dW{layer_idx}']
            self._param_dict[f'b{layer_idx}'] -= self._lr * grad_dict[f'db{layer_idx}']
    
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        
        train_loss, val_loss = [], []

        #in each epoch
        for _ in range(self._epochs):
            #run forward alg
            y_hat, cache = self.forward(X_train)
            #calc loss
            loss = self._binary_cross_entropy(y_train, y_hat) if self._loss_func == 'binary_cross_entropy' else self._mean_squared_error(y_train, y_hat)
            train_loss.append(loss)
            #run backward prop

            grad_dict = self.backprop(y_train, y_hat, cache)
            #update params for next epoch
            self._update_params(grad_dict)
        return train_loss, val_loss
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        y_hat, _ = self.forward(X)
        return y_hat
    
    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        A = self._sigmoid(Z)  # Compute sigmoid activation
        dZ = dA * A.T * (1 - A).T      # Compute gradient for backprop
        return dZ
  
    def _relu(self, Z: ArrayLike) -> ArrayLike:
        return np.maximum(0, Z)
    
    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        dZ = np.array(dA, copy=True)
        print(dZ)
        print(Z.T)
        dZ[Z.T <= 0] = 0
        return dZ
    
    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        m = y.shape[0]
        return -np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)) / m
    
    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        return -(np.divide(y, y_hat + 1e-9) - np.divide(1 - y, 1 - y_hat + 1e-9))
    
    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        return np.mean(np.power(y - y_hat, 2))
    
    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        return 2 * (y_hat - y) / y.shape[0]
    


# ======= SIMPLE AUTOENCODER ARCHITECTURE =======
autoencoder_arch = [
    {"input_dim": 3, "output_dim": 2, "activation": "relu"},  # Encoder
    {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"}  # Decoder
]

# ======= CREATE A SIMPLE NEURAL NETWORK INSTANCE =======
autoencoder = NeuralNetwork(
    nn_arch=autoencoder_arch,
    lr=0.01,
    seed=42,
    batch_size=1,
    epochs=1,
    loss_function="mean_squared_error"
)

"""
# ======= DEFINE TEST INPUT DATA =======
X_test = np.array([[0.1, 0.5, 0.9]])  # Single test sample
y_test = np.array([[0.2, 0.6, 0.8]])  # Target output

# ======= TEST FORWARD PASS =======
print("\n=== Testing Forward Pass ===")
output, cache = autoencoder.forward(X_test)
print(f"Output of Forward Pass: {output}")

print(cache)

# ======= TEST BACKPROPAGATION =======
print("\n=== Testing Backpropagation ===")
gradients = autoencoder.backprop(y_test, output, cache)

# ======= PRINT GRADIENTS =======
for key in gradients:
    print(f"{key} shape: {gradients[key].shape}")

# ======= UPDATE PARAMETERS =======
print("\n=== Testing Parameter Update ===")
autoencoder._update_params(gradients)

# ======= PREDICT FUNCTION =======
print("\n=== Testing Prediction ===")
y_pred = autoencoder.predict(X_test)
print(f"Predicted Output: {y_pred}")
"""