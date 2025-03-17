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
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
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
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
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
       

        dW_curr = np.dot(dZ_curr, A_prev.T) / A_prev.shape[1]
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / A_prev.shape[1]
        print(W_curr.T.shape)
        print(dZ_curr.shape)
        dA_prev = np.dot(W_curr.T, dZ_curr)
        
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        grad_dict = {}
        if self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        #backprop over layers of architecture, finding gradient and updating deltas
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']

            Z_curr = cache[f'Z{layer_idx}']
            A_prev = cache[f'A{layer_idx-1}'] if layer_idx > 1 else y

            dA_curr, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, layer['activation'])
            grad_dict[f'dW{layer_idx}'] = dW_curr
            grad_dict[f'db{layer_idx}'] = db_curr
        
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
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
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
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

    
    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        sig = self._sigmoid(Z).T
        return dA  * (1 - sig)


    def _relu(self, Z: ArrayLike) -> ArrayLike:
        return np.maximum(0, Z)
    
    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
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
    


nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}, {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
sample_nn = NeuralNetwork(nn_arch = nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="binary_cross_entropy")
W = np.array([[0.2, 0.3], [0.4, 0.5]])
b = np.array([[0.1], [0.2]])
A_prev = np.array([[1, 2], [3, 4]])  # Input features
    
A, Z = sample_nn._single_forward(W, b, A_prev, "relu")
print(A)
print(Z)


X = np.array([[0.5, 1.5]])
y_hat, cache = sample_nn.forward(X)

print(X)
print(y_hat)
print(cache)