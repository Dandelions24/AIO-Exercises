#E1. 
tp = 2
def calculate_precision_recall_f1_score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

# Example usage
tp = 2
fp = 3
fn = 4

precision, recall, f1_score = calculate_precision_recall_f1_score(tp, fp, fn)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)


#E2.
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def relu(x):
    return max(0, x)
def elu(x, alpha=1.0):
    return x if x >= 0 else alpha * (math.exp(x) - 1)
x = 3
print("Sigmoid:", sigmoid(x))
print("ReLU:", relu(x))
print("ELU:", elu(x))

#E3.
import math as np

def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred)**2)
    return mse

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


#E4.

import math

def sin_approx(x, n):
    sin_x = 0
    for i in range(n):
        sin_x = ((-1)**i) * (x**(2*i + 1)) / math.factorial(2*i + 1)
    return sin_x

def cos_approx(x, n):
    cos_x = 0
    for i in range(n):
        cos_x = ((-1)**i) * (x**(2*i)) / math.factorial(2*i)
    return cos_x

def sinh_approx(x, n):
    sinh_x = 0
    for i in range(n):
        sinh_x = x**(2*i + 1) / math.factorial(2*i + 1)
    return sinh_x

def cosh_approx(x, n):
    cosh_x = 0
    for i in range(n):
        cosh_x = x**(2*i) / math.factorial(2*i)
    return cosh_x

# Input values
x = 1.5
n = 5

# Compute approximations
sin_x_approx = sin_approx(x, n)
cos_x_approx = cos_approx(x, n)
sinh_x_approx = sinh_approx(x, n)
cosh_x_approx = cosh_approx(x, n)

# Print results
print(f"sin({x}) ≈ {sin_x_approx}")
print(f"cos({x}) ≈ {cos_x_approx}")
print(f"sinh({x}) ≈ {sinh_x_approx}")
print(f"cosh({x}) ≈ {cosh_x_approx}")


#E5.
import math as np

def mean_difference_nth_root_error(true_values, approx_values, n):
    true_values = np.array(true_values)
    approx_values = np.array(approx_values)  
    difference = np.abs(true_values**(1/n) - approx_values**(1/n))
    mdnre = np.mean(difference)
    return mdnre

true_values = [27, 64, 125]
approx_values = [25, 63, 120]
n = 3

mdnre = mean_difference_nth_root_error(true_values, approx_values, n)
print("Mean Difference of nth Root Error:", mdnre)


