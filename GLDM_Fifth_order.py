#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import psutil
import os
deg=5
case_name='NDVI'
class Errors:
    def __init__(self):
        self.E = 0.0
        self.D = 0.0
        self.minFH = 0  # reasonable forecasting horizon

class Solution:
    def __init__(self):
        self.a = None
        self.z = None
        self.Z = 0.0
        self.py = None

# Constants
LARGE = 0x10000

# Initialize the G array with None values for placeholders
G = [None] * 21  # Adjusted to have 20 coefficients + 1 for indexing consistency

# Define G functions for a fifth-order model considering up to 5 variables
# and their interactions that fit within the 20-coefficient constraint

# Single variables
def G1(x1, x2, x3, x4, x5): return x1
def G2(x1, x2, x3, x4, x5): return x2
def G3(x1, x2, x3, x4, x5): return x3
def G4(x1, x2, x3, x4, x5): return x4
def G5(x1, x2, x3, x4, x5): return x5

# Square terms for each variable
def G6(x1, x2, x3, x4, x5): return x1**2
def G7(x1, x2, x3, x4, x5): return x2**2
def G8(x1, x2, x3, x4, x5): return x3**2
def G9(x1, x2, x3, x4, x5): return x4**2
def G10(x1, x2, x3, x4, x5): return x5**2

# Selected two-variable interactions
def G11(x1, x2, x3, x4, x5): return x1 * x2
def G12(x1, x2, x3, x4, x5): return x1 * x3
def G13(x1, x2, x3, x4, x5): return x1 * x4
def G14(x1, x2, x3, x4, x5): return x1 * x5
def G15(x1, x2, x3, x4, x5): return x2 * x3

# Additional interactions or higher-order terms as needed
# Placeholder functions for the remaining coefficients
# These should be defined based on the specific model needs, ensuring we do not exceed 20 coefficients
def G16(x1, x2, x3, x4, x5): return x2 * x4
def G17(x1, x2, x3, x4, x5): return x3 * x4
def G18(x1, x2, x3, x4, x5): return x4 * x5
def G19(x1, x2, x3, x4, x5): return x1 * x2 * x3
def G20(x1, x2, x3, x4, x5): return x2 * x3 * x4

def GForming():
    # Initialize only the required G functions
    G[1] = G1
    G[2] = G2
    G[3] = G3
    G[4] = G4
    G[5] = G5
    G[6] = G6
    G[7] = G7
    G[8] = G8
    G[9] = G9
    G[10] = G10
    G[11] = G11
    G[12] = G12
    G[13] = G13
    G[14] = G14
    G[15] = G15
    G[16] = G16
    G[17] = G17
    G[18] = G18
    G[19] = G19
    G[20] = G20

# Ensure to call GForming() early in your script to initialize the G functions
GForming()





def SSTForming(_Y):
    # Assuming summs_count is the number of G functions, which is 20 in this context
    summs_count = 20
    # Initialize the SST matrix with dimensions [summs_count + 1] x [summs_count + 1]
    # We add 1 to each dimension to account for 1-based indexing used in the loops
    _SST = [[0.0 for _ in range(summs_count + 1)] for _ in range(summs_count + 1)]

    # Fill the SST matrix based on the G functions and the input series _Y
    # Note: Adjust the range of k as needed based on the input series length and the requirement of the G functions
    for i in range(1, summs_count + 1):
        for j in range(1, summs_count + 1):
            for k in range(6, len(_Y)):  # Assuming _Y[0] is not used or is a placeholder
                # Calculate the matrix element values using the appropriate G functions
                _SST[i][j] += G[i](_Y[k - 1], _Y[k - 2], _Y[k - 3], _Y[k - 4], _Y[k - 5]) * \
                              G[j](_Y[k - 1], _Y[k - 2], _Y[k - 3], _Y[k - 4], _Y[k - 5])

    # Print the SST matrix for verification/debugging
    print('\nMatrix SST\n')
    for i in range(1, summs_count + 1):
        for j in range(1, summs_count + 1):
            # Print each element with formatting for readability
            print(f"{_SST[i][j]:.4f}\t", end='')
        print()  # Newline at the end of each row for proper formatting

    return _SST



def JGTransforming(nn, _SST):
    # Dynamically adapt to the dimensions of _SST
    actual_nn = min(nn, len(_SST) - 1)  # Adjust based on the actual row count of _SST
    # Correcting the calculation of max_column_index to ensure it reflects the intended column range
    max_column_index = min(len(_SST[0]), nn + 1)  # Corrected to consider the actual column count correctly

    for iter_first in range(1, actual_nn + 1):
        mm = iter_first
        M = abs(_SST[iter_first][iter_first])

        for iter_second in range(iter_first + 1, actual_nn + 1):
            Mi = abs(_SST[iter_second][iter_first])
            if Mi > M:
                mm = iter_second
                M = Mi

        # Swap the rows
        _SST[iter_first], _SST[mm] = _SST[mm], _SST[iter_first]

        # Normalize the current pivot row
        Temp = _SST[iter_first][iter_first]
        for iter_second in range(iter_first, max_column_index):
            _SST[iter_first][iter_second] /= Temp

        # Eliminate the current column from all other rows
        for iter_second in range(1, actual_nn + 1):
            if iter_second != iter_first:
                Temp = _SST[iter_second][iter_first]
                for iter_third in range(iter_first, max_column_index):
                    _SST[iter_second][iter_third] -= _SST[iter_first][iter_third] * Temp

    # Optional: Print the transformed matrix for debugging
    print('\nTransformed Matrix:\n')
    for i in range(1, actual_nn + 1):
        for j in range(1, max_column_index):
            print(f"{_SST[i][j]:.4f}\t", end='')
        print()  # New line at the end of each row




def P1Forming(_Y, _SST):
    summs_count = 20  # Assuming 20 G functions
    t_max = len(_Y)  # The maximum index for _Y
    _P1 = [[0.0 for _ in range(summs_count + 1)] for _ in range(t_max)]
    
    for t in range(6, t_max):  # Start from 6 to ensure _Y[t - 5] is valid
        for j in range(1, summs_count + 1):
            for k in range(1, summs_count + 1):  # Ensure k iterates within bounds
                # Check bounds for _SST access
                if k < len(_SST) and summs_count + j < len(_SST[k]):
                    _P1[t][j] += G[k](_Y[t - 1], _Y[t - 2], _Y[t - 3], _Y[t - 4], _Y[t - 5]) * _SST[k][summs_count + j]
                else:
                    print(f"Access to _SST[{k}][{summs_count + j}] is out of bounds.")
    
    return _P1



def PForming(_Y, _P1):
    # Assuming impl_len is correctly set to the length of _Y minus 1 for 1-based indexing
    # Initialize the _P matrix with dimensions [impl_len + 2] x [impl_len + 2]
    _P = [[0.0 for _ in range(impl_len + 2)] for _ in range(impl_len + 2)]

    # Loop starts from 6 to accommodate the fifth order (requiring 5 previous values)
    for iter_first in range(6, impl_len + 1):
        for iter_second in range(6, impl_len + 1):
            # Reset the value before the subtraction loop
            _P[iter_first][iter_second] = 0.0

            # Ensure iter_third does not exceed the defined G functions
            for iter_third in range(1, min(summs_count + 1, len(G))):
                # Check bounds for _Y access
                if iter_second - 5 > 0:  # Ensure there are enough previous values
                    _P[iter_first][iter_second] -= G[iter_third](
                        _Y[iter_second - 1], _Y[iter_second - 2],
                        _Y[iter_second - 3], _Y[iter_second - 4], _Y[iter_second - 5]
                    ) * _P1[iter_first][iter_third]

            # Add 1 to the diagonal element
            if iter_first == iter_second:
                _P[iter_first][iter_first] += 1.0

    # Printing the matrix with updates to handle potential out-of-range issues
    print('\nMatrix P[6:m][6:m]\n')
    for iter_first in range(6, impl_len + 1):
        print('\n', iter_first, '\t', end='')
        for iter_third in range(6, impl_len + 1):
            print(f"{_P[iter_first][iter_third]:.4f}\t", end='')

    return _P


def PrGradForming(_Y, _P):
    # Initialize the _Prgrad array
    _Prgrad = [0.0 for _ in range(impl_len + 2)]
    _grad = [0.0 for _ in range(impl_len + 2)]

    # Copying _Y values to _grad
    for i in range(1, impl_len + 2):
        _grad[i] = _Y[i]

    # Start from 6 for the fifth-order model
    for iter_first in range(6, impl_len + 1):
        _Prgrad[iter_first] = 0.0
        for iter_second in range(6, impl_len + 1):
            _Prgrad[iter_first] += _P[iter_first][iter_second] * _grad[iter_second]

    # Printing the results
    print('\ni   grad[i]   Prgrad[i]    p[i]  \n', end='')
    for iter_first in range(6, impl_len + 1):
        print(f'\n{iter_first}\t{_grad[iter_first]}\t{_Prgrad[iter_first]}\t', end='')

    return _Prgrad


def DualWLDMSolution(_w, _p, _Prgrad):
    Al = LARGE
    Alc = 0

    # Start from 6 for the fifth-order model
    for iter_first in range(6, impl_len + 1):
        _w[iter_first] = 0

    iter_first = 6  # Initialize iter_first to 6 for the fifth-order model
    while iter_first < impl_len - summs_count - 2:
        Al = LARGE
        for iter_second in range(6, impl_len + 1):  # Start loop from 6
            if abs(_w[iter_second]) == _p[iter_second]:
                continue
            else:
                if _Prgrad[iter_second] > 0:
                    Alc = (_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]
                elif _Prgrad[iter_second] < 0:
                    Alc = (-_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]

                if Alc < Al:
                    Al = Alc

        for iter_second in range(6, impl_len + 1):  # Continue loop from 6
            if abs(_w[iter_second]) != _p[iter_second]:
                _w[iter_second] += Al * _Prgrad[iter_second]
                if abs(_w[iter_second]) == _p[iter_second]:
                    iter_first += 1







def PrimalWLDMSolution(_Y, _SST, _w, _p, _a, _z):
    lc_r = [0 for _ in range(summs_count + 1)]  # Ensure this is adequately sized
    lc_ri = 0  # The amount of basic equations of the primal problem
    
    # Adjusted loop to start from 1 if your indexing elsewhere assumes 1-based indexing for _Y
    for iter_first in range(5, impl_len + 1):
        if abs(_w[iter_first]) != _p[iter_first]:
            if lc_ri < len(lc_r):  # Removed -1 to properly utilize the allocated space in lc_r
                lc_r[lc_ri] = iter_first
                lc_ri += 1
            else:
                print(f"Error: lc_ri ({lc_ri}) exceeded lc_r bounds.")
                break  # or adjust as needed

    # Ensure iter_second does not exceed the bounds of initialized G functions
    for iter_first in range(1, lc_ri):
        for iter_second in range(1, min(lc_ri, len(G))):  # Min between lc_ri and G length
            # Safely accessing _Y with bounds checking
            indices = [lc_r[iter_first] - x for x in range(1, 6)]
            if min(indices) >= 1 and max(indices) <= len(_Y):  # Adjusted bounds checking for _Y access
                _SST[iter_first][iter_second] = G[iter_second](*[_Y[idx] for idx in indices])
            else:
                print(f"Index out of bounds for _Y access: {indices}")

        _SST[iter_first][lc_ri] = _Y[lc_r[iter_first]]  # Corrected column index for assignment

    JGTransforming(lc_ri, _SST)

    # Assigning results to _a and _z based on transformed _SST
    for iter_first in range(1, lc_ri):
        _a[iter_first] = _SST[iter_first][lc_ri]
        _z[lc_r[iter_first]] = 0



def GLDMEstimator(_Y):
    impl_len = len(_Y) - 1  # Assuming _Y[0] is unused or a placeholder, for 1-based indexing
    summs_count = 20  # Update this to match the actual number of G functions used

    # Initialize weights, parameters, and approximation errors
    lc_w = [0.0 for _ in range(impl_len + 2)]
    lc_p = [1.0 for _ in range(impl_len + 2)]
    lc_a1 = [0.0 for _ in range(summs_count + 1)]
    lc_a = [0.0 for _ in range(summs_count + 1)]
    lc_z = [0.0 for _ in range(impl_len + 2)]

    # Form matrices and projection vectors
    lc_SST = SSTForming(_Y)
    JGTransforming(summs_count, lc_SST)
    lc_P1 = P1Forming(_Y, lc_SST)
    lc_P = PForming(_Y, lc_P1)
    lc_Prgrad = PrGradForming(_Y, lc_P)

    Z = d = 0.0
    while True:
        # Update weights and parameters based on current approximation errors
        for i in range(1, summs_count + 1):
            lc_a1[i] = lc_a[i]

        for i in range(1, impl_len + 1):
            lc_p[i] = 1.0 / (1.0 + lc_z[i] ** 2)

        # Reset weights
        for i in range(1, impl_len + 1):
            lc_w[i] = 0.0

        # Solve the dual and primal problems
        DualWLDMSolution(lc_w, lc_p, lc_Prgrad)
        PrimalWLDMSolution(_Y, lc_SST, lc_w, lc_p, lc_a, lc_z)

        # Reset approximation errors and calculate new errors based on updated parameters
        Z = 0.0  # Reinitialize Z for sum of absolute differences
        for i in range(6, impl_len + 1):
            lc_z[i] = _Y[i] - sum([lc_a[j] * G[j](_Y[i - 1], _Y[i - 2], _Y[i - 3], _Y[i - 4], _Y[i - 5]) for j in range(1, summs_count + 1)])
            Z += abs(lc_z[i])

        # Check for convergence
        d = max([abs(lc_a[i] - lc_a1[i]) for i in range(1, summs_count + 1)])
        if d < 0.5:  # Adjust tolerance as needed
            break

    # Construct and return the solution
    Sol = Solution()
    Sol.a = lc_a
    Sol.z = lc_z
    Sol.Z = Z

    return Sol


def ForecastingEst(Y, Sol):
    # Adjust initialization of PY to ensure it's properly sized for operations
    PY = [[0.0 for _ in range(len(Y))] for _ in range(len(Y))]
    FH = [0 for _ in range(len(Y))]
    e = Errors()

    # Adjusting for fifth-order, ensure initial conditions are properly set
    for St in range(len(Y) - 5):  # Adjust to leave room for fifth-order initial conditions
        # Setting up initial conditions for fifth-order
        PY[St][0] = Y[St] if St < len(Y) else 0
        PY[St][1] = Y[St + 1] if St + 1 < len(Y) else 0
        PY[St][2] = Y[St + 2] if St + 2 < len(Y) else 0
        PY[St][3] = Y[St + 3] if St + 3 < len(Y) else 0
        PY[St][4] = Y[St + 4] if St + 4 < len(Y) else 0  # New initial condition for fifth order
        t = 5  # Start forecasting from the sixth element, accounting for fifth-order

        while True:
            # Break if 't' or 'St + t' goes beyond the bounds of Y
            if St + t >= len(Y):
                break

            py = 0
            for j in range(1, len(Sol.a)):
                # Ensure we're not accessing beyond the bounds of PY[St]
                if t - 5 >= 0:  # Adjusted to use five previous values
                    A1 = G[j](PY[St][t - 1], PY[St][t - 2], PY[St][t - 3], PY[St][t - 4], PY[St][t - 5])  # Adjust G function call
                    py += Sol.a[j] * A1
                else:
                    break  # Break the loop if we don't have enough data for forecasting

            # Assign forecasted value
            PY[St][t] = py

            # Increment 't' for the next forecasting step
            t += 1

            # Break if the forecast error exceeds the tolerance
            if abs(PY[St][t - 1] - Y[St + t - 1]) > Sol.Z:
                break

        # Record the forecasting horizon
        FH[St] = t - 1

    # Calculate minimum forecasting horizon
    e.minFH = min(FH)

    # Calculate errors for the forecasting horizon
    e.E, e.D = 0, 0
    for St in range(len(Y) - 5):  # Adjust range to account for fifth order
        for t in range(6, e.minFH + 1):  # Start error calculation from the sixth element
            if St + t < len(Y):
                e.D += abs(Y[St + t] - PY[St][t])
                e.E += (Y[St + t] - PY[St][t])

    if e.minFH > 0:  # Avoid division by zero
        e.E /= e.minFH
        e.D /= e.minFH

    return e



def calculate_rmse(actual, predicted):
    return np.sqrt(((np.array(actual) - np.array(predicted)) ** 2).mean())

def calculate_r_squared(actual, predicted):
    correlation_matrix = np.corrcoef(actual, predicted)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    return r_squared

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    non_zero_actual = actual != 0
    return np.mean(np.abs((actual[non_zero_actual] - predicted[non_zero_actual]) / actual[non_zero_actual])) * 100


def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def calculate_mse(actual, predicted):
    return np.mean((np.array(actual) - np.array(predicted)) ** 2)

def calculate_me(actual, predicted):
    return np.mean(np.array(actual) - np.array(predicted))

def calculate_median_absolute_error(actual, predicted):
    return np.median(np.abs(np.array(actual) - np.array(predicted)))


def calculate_mase(actual, predicted, seasonal_period=1):
    actual, predicted = np.array(actual), np.array(predicted)
    n = len(actual)
    d = np.abs(np.diff(actual, n=seasonal_period)).sum() / (n - seasonal_period)
    errors = np.mean(np.abs(actual - predicted))
    return errors / d

def calculate_mbe(actual, predicted):
    return np.mean(np.array(actual) - np.array(predicted))

def calculate_time_series_values(Y, Sol, length):
    """
    Calculate time series values using the coefficients in Solution for a fifth-order model.

    Parameters:
    Y (list): The initial values of the time series.
    Sol (Solution): The solution object containing the coefficients.
    length (int): The number of time series values to calculate.

    Returns:
    list: Calculated time series values.
    """
    calculated_values = [0.0 for _ in range(length)]

    # Assuming the first five values of Y are initial values for a fifth-order model
    for i in range(5):
        calculated_values[i] = Y[i]

    # Calculate the rest of the values based on the coefficients
    for t in range(5, length):
        value = Sol.a[0]  # This could be an intercept if your model has one
        for i in range(1, len(Sol.a)):
            value += Sol.a[i] * G[i](Y[t - 1], Y[t - 2], Y[t - 3], Y[t - 4], Y[t - 5])
        calculated_values[t] = value

    return calculated_values



def plot_time_series(original_data, calculated_data, series_number, save_path):
    plt.figure(figsize=(10, 6))

    # Create a new x-axis range that starts from 1
    x_axis_range = range(1, len(original_data) + 1)

    # Use the new x-axis range for plotting
    plt.plot(x_axis_range, original_data, label='Original', color='blue', linewidth=2)
    plt.plot(x_axis_range, calculated_data, label='GLDM Model', color='black', linestyle='dotted', linewidth=2)

    plt.title(f'Time Series {case_name}: Original vs GLDM Model', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)

    # Automatically place the legend in the best location
    plt.legend(loc='best', fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot to a file
    plt.savefig(save_path, format='png', dpi=300)

    # Optionally display the plot
    # plt.show()

    # Close the plot to free memory
    plt.close()


def plot_time_series_adjusted(original_data, calculated_data, series_number, save_path):
    plt.figure(figsize=(10, 6))

    # Ensure the first four values of calculated_data match the original_data
    if len(original_data) >= 6 and len(calculated_data) >= 6:
        for i in range(6):  # Update this loop to copy the first four values
            calculated_data[i] = original_data[i]

    # Adjust lengths if necessary to ensure both lists are of equal length
    min_length = min(len(original_data), len(calculated_data))

    # Modify here to remove the first and last value from both sets of data
    original_data_adjusted = original_data[1:min_length-1]  # Exclude the first and last items
    calculated_data_adjusted = calculated_data[1:min_length-1]  # Exclude the first and last items

    # Adjust the x-axis range to start from 1 after removing the first and last values
    # The new range needs to reflect the reduced number of data points
    x_axis_range = range(1, min_length - 1)  # Adjusted to start from 1 and match the new length
 
    # Plotting the original and calculated data
    plt.plot(x_axis_range, original_data_adjusted, label='Original', color='blue', linewidth=2, marker='o') 
    plt.plot(x_axis_range, calculated_data_adjusted, label='GLDM Model', color='red', linestyle='dotted', linewidth=2, marker='o')

    # Setting the plot title and labels
    plt.title(f'Time Series {case_name}: Original vs GLDM Model', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)

    # Adding a legend and grid
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Saving the plot to a file and closing the plot figure
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()




def main():
    # Start measuring time and resources
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory_use = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    # File handling
    with open("input.txt", "r") as f, open("output.txt", "w") as g:       
        # Data Input
        lc_c = ''
        while lc_c != ':':
            lc_c = f.read(1)
        m, ts = map(int, f.readline().split())
        global impl_len
        impl_len = m  # Length of time series
        global summs_count
        summs_count = 31  # Updated to 31 for the fifth-order model
        print(f"Length: {m}\nTime series: {ts}\n")
        
        # Reading time series data
        setnum = 0
        RY = [[] for _ in range(ts)]
        for i in range(ts):
            RY[i] = [0.0] * (m + 2)
        while setnum < ts:
            ic = 1
            while ic <= m:
                line = f.readline()
                s = float(line)
                RY[setnum][ic] = s
                ic += 1
            setnum += 1
        GL_RY = RY
        
        # Writing results to a file
        g.write(f"Number of time series: {ts}\n")
        g.write(f"Length of time series: {impl_len}\n")

        # Processing each time series
        for sn in range(ts):
            Y = [0.0] * (m + 2)
            for j in range(1, m + 1):
                Y[j] = RY[sn][j]
            GL_Y = GL_RY[sn]

            lc_SST = np.zeros((summs_count + 1, summs_count * 2 + 2), dtype=float)
            GForming()
            lc_SST = SSTForming(GL_Y)
            JGTransforming(summs_count, lc_SST)
            Sol = GLDMEstimator(GL_Y)
            print("GLDMEstimator() OK\n")


        # Calculate the time series values using the obtained coefficients
            calculated_ts_values = calculate_time_series_values(GL_Y, Sol, len(GL_Y))
            # Error calculations and table display
            original_data_trimmed = GL_Y  # Keep original data as is
            calculated_data_trimmed = calculated_ts_values [:]  # Ignore last two values from calculated data
            # Assuming the first value is manually set and should be excluded from error calculations
            start_index = 6 # Change to 2 if you need to skip the first two values for some reason

            # Ensure the lengths match
            min_length = min(len(original_data_trimmed), len(calculated_ts_values))
            original_data_trimmed = original_data_trimmed[:min_length]
            calculated_data_trimmed = calculated_ts_values[:min_length]

            # Calculate errors using consistent slicing
            mae = calculate_mae(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            mbe = calculate_mbe(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            mse = calculate_mse(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            rmse = calculate_rmse(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            r_squared = calculate_r_squared(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            mape = calculate_mape(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            me = calculate_me(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])
            median_abs_error = calculate_median_absolute_error(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1])

            # Assuming a seasonal period for MASE calculation; adjust as necessary
            seasonal_period = 1  # This should be set based on your data's seasonality
            mase = calculate_mase(original_data_trimmed[start_index:-1], calculated_data_trimmed[start_index:-1], seasonal_period)

           # Write the results to the file
            g.write(f"Error Matrix start from six point to the end of dataset\n")
            g.write(f"RMSE: {rmse}\n")
            g.write(f"R-squared: {r_squared}\n")
            g.write(f"MAPE: {mape}\n")
            # Write the results to the file
            g.write(f"MAE: {mae}\n")
            g.write(f"MSE: {mse}\n")
            g.write(f"ME: {me}\n")
            g.write(f"Median Absolute Error: {median_abs_error}\n")
            # Write the results to the file
            g.write(f"MASE: {mase}\n")
            g.write(f"MBE: {mbe}\n")
            # Prepare and display the table
            g.write(f"{'Original Data':<20}{'Calculated Data':<20}{'Error':<20}\n")
            for orig, calc in zip(original_data_trimmed, calculated_data_trimmed):
                error = orig - calc
                g.write(f"{orig:<20}{calc:<20}{error:<20}\n")

           # Write model coefficients and errors
            ANS = [0] * (summs_count + 2)
            ANS[0] = sn  # Time Series Number
            ANS[1] = 0   # Placeholder for future use or additional data

            # Assigning model coefficients to ANS
            for i in range(21):
                ANS[i + 2] = Sol.a[i]  # Model Coefficients

            e = ForecastingEst(GL_Y, Sol)  # Forecasting Errors
            print("ForecastingEST OK\n")
            print(e.minFH, "\n", end='')
            ANS[21] = e.minFH  # Minimum Forecasting Horizon
            ANS[22] = e.D     # Average Absolute Error
            ANS[23] = e.E     # Average Error
            ANS[24] = Sol.Z   # Sum of Absolute Differences between Model and Actual Data

            # Writing the results with descriptive labels
            g.write(f"Time Series Number: {ANS[0]}\n")
            g.write("Model Coefficients:\n")
            for i in range(1, 21):  # Assuming 20 coefficients
                g.write(f"Coefficient a{i}: {ANS[i]}\n")

            g.write(f"Minimum Forecasting Horizon: {ANS[21]}\n")
            g.write(f"Average Absolute Error: {ANS[22]}\n")
            g.write(f"Average Error: {ANS[23]}\n")
            g.write(f"Sum of Absolute Differences: {ANS[24]}\n")


            # Calculating and Writing G function values
            g.write("\nG Function Values:\n")
            if len(Y) >= 5:  # Ensure there are enough data points for fifth-order
                x1, x2, x3, x4, x5 = Y[-5], Y[-4], Y[-3], Y[-2], Y[-1]  # Last five values from the time series
                for i in range(1, 21):  # Corrected to match the actual count of G functions
                    try:
                        g_value = G[i](x1, x2, x3, x4, x5)
                        g.write(f"G{i}({x1}, {x2}, {x3}, {x4}, {x5}): {g_value}\n")
                    except IndexError:
                        print(f"Error accessing G function at index {i}. Ensure G is correctly initialized.")


         # Writing the original time series data
            g.write("Original Time Series Data:\n")
            for j in range(1, m+1):
                g.write(f"{Y[j]}\n")
            # Write original data trimmed
            g.write("original data trimmed:\n")
            for value in original_data_trimmed[start_index:-1]:
                g.write(f"{value}\n")
            # Define column widths, adjust as needed based on expected data width
            col_widths = [25, 25, 25]

            # Write titles (headers) for each column with consistent spacing for a table-like display
            headers = ["Original Data", "Calculated Data", "Error"]
            header_line = "".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
            g.write(header_line + "\n")
            g.write("-" * sum(col_widths) + "\n")  # Divider line for visual separation

            # Assuming original_data_trimmed, calculated_data_trimmed, and start_index are defined
            # Calculate the length of data to be iterated over
            data_length = min(len(original_data_trimmed) - start_index - 1, len(calculated_data_trimmed) - start_index - 1)

            # Iterate over the range of data_length to access each element by its index
            for i in range(data_length):
                original = original_data_trimmed[start_index + i]  # Accessing the original data value
                calculated = calculated_data_trimmed[start_index + i]  # Accessing the calculated data value
                error = original - calculated  # Calculating the error between the original and calculated values
                
                # Writing the data and error to the file without rounding
                # Convert numbers to strings directly
                original_str = f"{original}"
                calculated_str = f"{calculated}"
                error_str = f"{error}"
                
                # Format the line with consistent spacing for a table-like display
                data_line = f"{original_str:<{col_widths[0]}}{calculated_str:<{col_widths[1]}}{error_str:<{col_widths[2]}}"
                g.write(data_line + "\n")

 


            original_data = Y[1:-1]  # Adjust indices as per your data
            calculated_data = calculate_time_series_values(GL_Y, Sol, len(GL_Y))[1:-1]

            # Define a path for saving the plot
            plot_save_path = f"plot_series_{sn}.png"  # This will save the plot in the current directory
    

            # Call the plotting function with the save path
            plot_time_series(original_data, calculated_data, sn, plot_save_path)
            plot_time_series_adjusted(Y, calculated_ts_values, sn, f'plot_Adjusted_{sn}.png')
            # Stop measuring time and resources
            end_time = time.time()
            final_memory_use = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB


            # Stop measuring time and resources
            end_time = time.time()
            final_memory_use = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

            # Calculate total time and memory used
            total_time = end_time - start_time
            total_memory_used = final_memory_use - initial_memory_use
            g.write("\nPerformance Metrics:\n")
            g.write(f"Total Execution Time: {total_time:.2f} seconds\n")
            g.write(f"Total Additional Memory Used: {total_memory_used:.2f} MB\n")


    # Wait for user input before closing (simulating 'Press any key' behavior)
    input("\nPress any key: ")

if __name__ == "__main__":
    main()

