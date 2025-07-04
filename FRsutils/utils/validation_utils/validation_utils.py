import numpy as np

ALLOWED_RANKING_STRATEGIES = {'pos', 'lower', 'upper'}

def _validate_string_param_choice(param_name: str, 
                                  param_value: str, 
                                  allowed: set[str]) -> str:
    """
    @brief Validates a string parameter against allowed values.

    @param name Parameter name (for error reporting).
    @param value Actual user-provided value.
    @param allowed Allowed set of values.

    @return The same value if valid.

    @throws ValueError If value is not allowed.
    """
    if param_value not in allowed:
        raise ValueError(
            f"Invalid value '{param_value}' for parameter '{param_name}'. "
            f"Allowed values are: {sorted(allowed)}."
        )
    return param_value

def validate_ranking_strategy_choice(name: str):
    """
    @brief Validates a ranking strategy choice.
    """
    return _validate_string_param_choice("ranking_strategy", 
                                         name, 
                                         ALLOWED_RANKING_STRATEGIES)

# TODO: check correctness
def validate_range_0_1(x, name="name_value"):
    
    if isinstance(x, float):
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"{name} must be in range [0.0, 1.0], but got {x}")
    elif isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"{name} must be an array of floats")
        if np.any(x < 0.0) or np.any(x > 1.0):
            raise ValueError(f"All elements of {name} must be in range [0.0, 1.0]")
    else:
        raise TypeError(f"{name} must be a float or a numpy.ndarray, but got {type(x).__name__}")

    return x
##############################################################################

def validate_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    @brief Validates a matrix to ensure it meets specific criteria.
    
    This function checks whether the input matrix is:
    - A square matrix (number of rows == number of columns)
    - All elements are in the range [0, 1]
    - The main diagonal is entirely set to 1.0

    @param matrix A NumPy 2D array (nxn) to validate.
    @throws TypeError if the input is not a 2D NumPy ndarray.
    @throws ValueError if the matrix is not square, contains values outside [0, 1], 
            or the diagonal is not all 1.0.
    """
    # Check if input is a 2D numpy ndarray
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise TypeError("Input must be a 2D NumPy ndarray.")

    rows, cols = matrix.shape

    # Check if the matrix is square
    if rows != cols:
        raise ValueError("Matrix must be square (same number of rows and columns).")

    # Check if all values are in the range [0, 1]
    if not np.all((0 <= matrix) & (matrix <= 1)):
        raise ValueError("All matrix values must be in the range [0, 1].")

    # Check if the main diagonal is all 1.0
    if not np.all(np.diag(matrix) == 1.0):
        raise ValueError("All diagonal elements must be 1.0.")
    
    return matrix


def validate_y_array(y: np.ndarray, expected_num_classes: int) -> np.ndarray:
    """
    @brief Validates a 1D class array for compatibility with scikit-learn and imbalanced-learn.

    This function ensures that the input class array:
    - Is a 1D NumPy ndarray
    - Contains only valid class labels (integers or strings)
    - Has no NaN or infinite values
    - Contains exactly the expected number of unique classes

    @param y A 1D NumPy ndarray representing class labels.
    @param expected_num_classes The expected number of unique classes.
    @throws TypeError if the input is not a 1D NumPy ndarray.
    @throws ValueError if the array contains invalid values or does not match expected class count.
    """
    # Check if input is a 1D NumPy ndarray
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise TypeError("Input must be a 1D NumPy ndarray.")

    # Check for NaNs or infinite values
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Class array must not contain NaN or infinite values.")

    # Check that labels are of acceptable type (integers or strings)
    if not np.issubdtype(y.dtype, np.integer) and not np.issubdtype(y.dtype, np.str_):
        raise ValueError("Class labels must be integers or strings.")

    # Check number of unique classes
    unique_classes = np.unique(y)
    if len(unique_classes) != expected_num_classes:
        raise ValueError(f"Expected {expected_num_classes} unique classes, found {len(unique_classes)}.")

    return y

