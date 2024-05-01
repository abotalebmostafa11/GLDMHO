#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import psutil
import os
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

# Function pointers for a fourth-order model with 14 G functions
G = [None] * 15  # Including G[0] placeholder for convenience, G[1] to G[14] are used

# Single variables
def G1(x1, x2, x3, x4): return x1
def G2(x1, x2, x3, x4): return x2
def G3(x1, x2, x3, x4): return x3
def G4(x1, x2, x3, x4): return x4

# Squared terms
def G5(x1, x2, x3, x4): return x1 * x1
def G6(x1, x2, x3, x4): return x2 * x2
def G7(x1, x2, x3, x4): return x3 * x3
def G8(x1, x2, x3, x4): return x4 * x4

# Two-variable products
def G9(x1, x2, x3, x4): return x1 * x2
def G10(x1, x2, x3, x4): return x1 * x3
def G11(x1, x2, x3, x4): return x1 * x4
def G12(x1, x2, x3, x4): return x2 * x3
def G13(x1, x2, x3, x4): return x2 * x4
def G14(x1, x2, x3, x4): return x3 * x4

def GForming():
    # Assigning the correct functions to the G array
    global G
    G[1], G[2], G[3], G[4] = G1, G2, G3, G4
    G[5], G[6], G[7], G[8] = G5, G6, G7, G8
    G[9], G[10], G[11], G[12] = G9, G10, G11, G12
    G[13], G[14] = G13, G14


def SSTForming(_Y):
    # Ensure summs_count is correctly set to 14 for the fourth-order model
    global summs_count  # Assuming summs_count is defined globally
    summs_count = 14  # Correct number of coefficients for a fourth-order model

    # Creating the _SST matrix with the correct dimensions
    _SST = [[0.0 for _ in range(summs_count * 2 + 2)] for _ in range(summs_count + 1)]

    # Iterating over the matrix to perform calculations
    for i in range(1, summs_count + 1):
        for j in range(1, summs_count + 1):
            # Ensure k starts from 5 to have four previous values available
            for k in range(5, impl_len + 1):
                # Update to include four previous values in the time series
                # Note: Adjust G function calls if they are 1-based indexed
                _SST[i][j] += G[i](_Y[k - 4], _Y[k - 3], _Y[k - 2], _Y[k - 1]) * \
                              G[j](_Y[k - 4], _Y[k - 3], _Y[k - 2], _Y[k - 1])

        # Setting specific matrix elements to 0.0 and 1.0 for the augmented matrix
        for j in range(1, summs_count + 1):
            _SST[i][summs_count + j] = 0.0
        _SST[i][summs_count + i] = 1.0

    # Optionally printing the matrix for verification/debugging
    print('\nMatrix SST\n')
    for i in range(1, summs_count + 1):
        print('\n', i, '\t', end='')
        for j in range(1, summs_count * 2 + 2):  # Ensure printing through the augmented part
            print(_SST[i][j], '\t', end='')

    return _SST


def JGTransforming(nn, _SST):
    for iter_first in range(1, nn + 1):
        # Find Lead Row
        mm = iter_first
        M = abs(_SST[iter_first][iter_first])

        for iter_second in range(iter_first + 1, nn + 1):
            Mi = abs(_SST[iter_second][iter_first])
            if Mi > M:
                mm = iter_second
                M = Mi

        # Swapping of current N-th and lead mm-th rows
        _SST[iter_first], _SST[mm] = _SST[mm], _SST[iter_first]

        # Normalization of the current row
        Temp = _SST[iter_first][iter_first]
        for iter_second in range(iter_first, 2 * nn + 1):
            _SST[iter_first][iter_second] /= Temp

        # Orthogonalize the Current Column
        for iter_second in range(1, iter_first):
            Temp = _SST[iter_second][iter_first]
            for iter_third in range(iter_first, 2 * nn + 1):
                _SST[iter_second][iter_third] -= _SST[iter_first][iter_third] * Temp

        for iter_second in range(iter_first + 1, nn + 1):
            Temp = _SST[iter_second][iter_first]
            for iter_third in range(iter_first, 2 * nn + 1):
                _SST[iter_second][iter_third] -= _SST[iter_first][iter_third] * Temp

        # Printing the matrix
        print('\nMatrix SST^-1\n')
        for iter_first in range(1, nn + 1):
            print('\n', iter_first, '\t', end='')
            for iter_third in range(1, nn + nn + 1):
                print(_SST[iter_first][iter_third], '\t', end='')

def P1Forming(_Y, _SST):
    _P1 = [[0.0 for _ in range(summs_count + 1)] for _ in range(impl_len + 2)]

    # Adjust the range to start from 5, as we need four previous values for fourth-order
    for t in range(5, impl_len + 1):
        for j in range(1, summs_count + 1):
            _P1[t][j] = 0.0

            for k in range(1, summs_count + 1):
                # Update to include four previous values in the calculation
                _P1[t][j] += G[k](_Y[t - 1], _Y[t - 2], _Y[t - 3], _Y[t - 4]) * _SST[k][summs_count + j]

    # Printing the matrix
    print('\nMatrix P1[5:m][1:n]\n')  # Update the range in the print statement
    for iter_first in range(5, impl_len + 1):
        print('\n', iter_first, '\t', end='')
        for iter_second in range(1, summs_count + 1):
            print(_P1[iter_first][iter_second], '\t', end='')

    return _P1

def PForming(_Y, _P1):
    # Initialize the _P matrix
    _P = [[0.0 for _ in range(impl_len + 2)] for _ in range(impl_len + 2)]

    # Adjust the range to start from 5 for a fourth-order model
    for iter_first in range(5, impl_len + 1):
        for iter_second in range(5, impl_len + 1):
            _P[iter_first][iter_second] = 0.0

            for iter_third in range(1, summs_count + 1):
                # Update to include four previous values in the calculation
                _P[iter_first][iter_second] -= G[iter_third](_Y[iter_second - 1], _Y[iter_second - 2], _Y[iter_second - 3], _Y[iter_second - 4]) * _P1[iter_first][iter_third]

            if iter_first == iter_second:
                _P[iter_first][iter_first] += 1.0

    # Printing the matrix
    print('\nMatrix P[5:m][5:m]\n')  # Update the range in the print statement
    for iter_first in range(5, impl_len + 1):
        print('\n', iter_first, '\t', end='')
        for iter_third in range(5, impl_len + 1):
            print(_P[iter_first][iter_third], '\t', end='')

    return _P

def PrGradForming(_Y, _P):
    # Initialize the _Prgrad array
    _Prgrad = [0.0 for _ in range(impl_len + 2)]
    _grad = [0.0 for _ in range(impl_len + 2)]

    # Copying _Y values to _grad
    for i in range(1, impl_len + 2):
        _grad[i] = _Y[i]

    # Start from 5 for the fourth-order model
    for iter_first in range(5, impl_len + 1):
        _Prgrad[iter_first] = 0.0
        for iter_second in range(5, impl_len + 1):
            _Prgrad[iter_first] += _P[iter_first][iter_second] * _grad[iter_second]

    # Printing the results
    print('\ni   grad[i]   Prgrad[i]    p[i]  \n', end='')
    for iter_first in range(5, impl_len + 1):
        print(f'\n{iter_first}\t{_grad[iter_first]}\t{_Prgrad[iter_first]}\t', end='')

    return _Prgrad


def DualWLDMSolution(_w, _p, _Prgrad):
    Al = LARGE
    Alc = 0

    # Start from 5 for the fourth-order model
    for iter_first in range(5, impl_len + 1):
        _w[iter_first] = 0

    iter_first = 5  # Initialize iter_first to 5 for the fourth-order model
    while iter_first < impl_len - summs_count - 2:
        Al = LARGE
        for iter_second in range(5, impl_len + 1):  # Start loop from 5
            if abs(_w[iter_second]) == _p[iter_second]:
                continue
            else:
                if _Prgrad[iter_second] > 0:
                    Alc = (_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]
                elif _Prgrad[iter_second] < 0:
                    Alc = (-_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]

                if Alc < Al:
                    Al = Alc

        for iter_second in range(5, impl_len + 1):  # Start loop from 5
            if abs(_w[iter_second]) != _p[iter_second]:
                _w[iter_second] += Al * _Prgrad[iter_second]
                if abs(_w[iter_second]) == _p[iter_second]:
                    iter_first += 1


def PrimalWLDMSolution(_Y, _SST, _w, _p, _a, _z):
    lc_r = [0 for _ in range(summs_count + 1)]  # Ensure this is adequately sized
    lc_ri = 0  # The amount of basic equations of the primal problem
    
    for iter_first in range(5, impl_len + 1):  # Adjusted range to match your usage
        if abs(_w[iter_first]) != _p[iter_first]:
            if lc_ri < len(lc_r) - 1:  # Check to prevent index out of range
                lc_ri += 1
                lc_r[lc_ri] = iter_first
            else:
                # Consider resizing lc_r or handling the error
                print(f"Error: lc_ri ({lc_ri}) exceeded lc_r bounds.")
                break  # or continue based on desired handling

    # Continue with the rest of your function...


    for iter_first in range(1, lc_ri + 1):
        for iter_second in range(1, lc_ri + 1):
            # Update to include four previous values in the calculation
            _SST[iter_first][iter_second] = G[iter_second](_Y[lc_r[iter_first] - 1], _Y[lc_r[iter_first] - 2], _Y[lc_r[iter_first] - 3], _Y[lc_r[iter_first] - 4])

        _SST[iter_first][lc_ri + 1] = _Y[lc_r[iter_first]]

    JGTransforming(lc_ri, _SST)

    for iter_first in range(1, lc_ri + 1):
        _a[iter_first] = _SST[iter_first][lc_ri + 1]
        _z[lc_r[iter_first]] = 0


def GLDMEstimator(_Y):
    lc_w = [0.0 for _ in range(impl_len + 2)]  # WLDM weights
    lc_p = [1.0 for _ in range(impl_len + 2)]  # GLDM weights

    lc_a1 = [0.0 for _ in range(summs_count + 1)]
    lc_a = [0.0 for _ in range(summs_count + 1)]  # Identified parameters
    lc_z = [0.0 for _ in range(impl_len + 2)]  # WLDM approximation errors

    lc_SST = SSTForming(_Y)  # Matrix for J-G transforming
    JGTransforming(summs_count, lc_SST)
    lc_P1 = P1Forming(_Y, lc_SST)  # It is used for P calculation
    lc_P = PForming(_Y, lc_P1)  # Projection matrix
    lc_Prgrad = PrGradForming(_Y, lc_P)  # Projection of the gradient

    Z = d = 0.0
    while True:
        for i in range(1, summs_count + 1):
            lc_a1[i] = lc_a[i]

        for i in range(1, impl_len + 1):
            lc_p[i] = 1.0 / (1.0 + lc_z[i] * lc_z[i])

        for i in range(1, impl_len + 1):
            lc_w[i] = 0.0

        DualWLDMSolution(lc_w, lc_p, lc_Prgrad)
        print("Dual ok")
        PrimalWLDMSolution(_Y, lc_SST, lc_w, lc_p, lc_a, lc_z)
        print("Primal ok")

        Z = lc_z[1] = lc_z[2] = lc_z[3] = lc_z[4] = 0.0  # Resetting first four values for fourth-order
        for i in range(5, impl_len + 1):
            lc_z[i] = _Y[i]
            for j in range(1, summs_count + 1):
                lc_z[i] -= lc_a[j] * G[j](_Y[i - 1], _Y[i - 2], _Y[i - 3], _Y[i - 4])
            Z += abs(lc_z[i])

        d = max([abs(lc_a[i] - lc_a1[i]) for i in range(1, summs_count + 1)])
        if d < 0.5:  # Define some_tolerance_value as per your requirements
            break

    # Constructing the solution
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

    # Adjusting for fourth-order, ensure initial conditions are properly set
    for St in range(len(Y) - 4):  # Ensure we don't go beyond Y's bounds for initial conditions
        PY[St][0] = Y[St] if St < len(Y) else 0
        PY[St][1] = Y[St + 1] if St + 1 < len(Y) else 0
        PY[St][2] = Y[St + 2] if St + 2 < len(Y) else 0
        PY[St][3] = Y[St + 3] if St + 3 < len(Y) else 0
        t = 4  # Start forecasting from the fifth element

        while True:
            # Break if 't' or 'St + t' goes beyond the bounds of Y
            if St + t >= len(Y):
                break

            py = 0
            for j in range(1, len(Sol.a)):
                # Ensure we're not accessing beyond the bounds of PY[St]
                if t - 4 >= 0:
                    A1 = G[j](PY[St][t - 1], PY[St][t - 2], PY[St][t - 3], PY[St][t - 4])
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
    for St in range(len(Y) - 4):
        for t in range(5, e.minFH + 1):
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
    Calculate time series values using the coefficients in Solution.

    Parameters:
    Y (list): The initial values of the time series.
    Sol (Solution): The solution object containing the coefficients.
    length (int): The number of time series values to calculate.

    Returns:
    list: Calculated time series values.
    """
    calculated_values = [0.0 for _ in range(length)]

    # Assuming the first four values of Y are initial values
    for i in range(4):
        calculated_values[i] = Y[i]

    # Calculate the rest of the values based on the coefficients
    for t in range(4, length):
        value = Sol.a[0]  # This could be an intercept if your model has one
        for i in range(1, len(Sol.a)):
            value += Sol.a[i] * G[i](Y[t - 1], Y[t - 2], Y[t - 3], Y[t - 4])
        calculated_values[t] = value

    return calculated_values



def plot_time_series(original_data, calculated_data, series_number, save_path):
    plt.figure(figsize=(10, 6))

    # Create a new x-axis range that starts from 1
    x_axis_range = range(1, len(original_data) + 1)

    # Use the new x-axis range for plotting
    plt.plot(x_axis_range, original_data, label='Original Data', color='blue', linewidth=2)
    plt.plot(x_axis_range, calculated_data, label='GLDM Model', color='red', linestyle='dotted', linewidth=2)

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
    if len(original_data) >= 5 and len(calculated_data) >= 5:
        for i in range(5):  # Update this loop to copy the first four values
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
        # Reading until ':' is encountered
        lc_c = ''
        while lc_c != ':':
            lc_c = f.read(1)
        m, ts = map(int, f.readline().split())
        global impl_len
        impl_len = m  # Length of time series
        global summs_count
        summs_count = 19  # Updated to 19 for the fourth-order model
        print(f"Length: {m}\nTime series: {ts}\n")
        # End of Data Input 
        # Reading time series data
        setnum = 0
        RY = [[] for _ in range(ts)]  # Create the ts arrays for time series
        for i in range(ts):
            RY[i] = [0] * (m + 2)
        while setnum < ts:
            print("Reading time series", setnum)
            ic = 1
            while ic <= m:
                line = f.readline()  # Read the next line
                s = float(line)
                RY[setnum][ic] = s
                ic += 1
            print("\n Finished reading of time series", setnum)
            setnum += 1
        GL_RY = RY        # End Reading time series data     
        
        # Writing results to a file
        g.write(f"Number of time series: {ts}\n")
        g.write(f"Length of time series: {impl_len}\n")

            
        
        # Processing each time series
        for sn in range(ts):
            Y = [0] * (m + 2)
            k = 1
            for j in range(1, m + 1):
                Y[j] = RY[sn][j]
                k += 1
                # cout<<Y[j]<<" ";
            GL_Y = GL_RY[sn]

            lc_SST = np.zeros((summs_count + 1, summs_count * 2 + 2), dtype=float)
            # Solution
            GForming()
            print("GForming() OK\n")
            # Assuming lc_SST, GForming, SSTForming, JGTransforming, GLDMEstimator, and ForecastingEst functions are defined
            lc_SST = SSTForming(GL_Y)
            JGTransforming(summs_count, lc_SST)
            print("\n JGTransforming() OK\n")
            Sol = GLDMEstimator(GL_Y)
            print("GLDMEstimator() OK\n")


            # Calculate the time series values using the obtained coefficients
            calculated_ts_values = calculate_time_series_values(GL_Y, Sol, len(GL_Y))
            # Error calculations and table display
            original_data_trimmed = GL_Y  # Keep original data as is
            calculated_data_trimmed = calculated_ts_values [:]  # Ignore last two values from calculated data
            # Assuming the first value is manually set and should be excluded from error calculations
            start_index = 5 # Change to 2 if you need to skip the first two values for some reason

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
            g.write(f"Error Matrix start from Fifth point to the end of dataset\n")
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
            # Write model coefficients and errors
            ANS = [0] * 25
            ANS[0] = sn  # Time Series Number
            ANS[1] = 0   # Placeholder for future use or additional data
            for i in range(summs_count):
                ANS[i + 2] = Sol.a[i + 1]  # Model Coefficients
            ANS[21] = Sol.Z  # Sum of Absolute Differences between Model and Actual Data

            e = ForecastingEst(GL_Y, Sol)  # Forecasting Errors
            print("ForecastingEST OK\n")
            print(e.minFH, "\n", end='')
            ANS[22] = e.minFH  # Minimum Forecasting Horizon
            ANS[23] = e.D     # Average Absolute Error
            ANS[24] = e.E     # Average Error

            # Adjusting to write only 14 coefficients correctly
            g.write(f"Time Series Number: {ANS[0]}\n")
            g.write("Model Coefficients:\n")

            # Loop through the coefficients. Note: ANS list starts coefficients from index 2 to 15 for 14 coefficients
            for i in range(2, 16):  # Adjusted to 16 because it's exclusive and you start at 2
                g.write(f"Coefficient a{i-1}: {ANS[i]:.4f}\n")  # Assuming ANS[i] stores coefficients and formatting for clarity

            g.write(f"Sum of Absolute Differences: {ANS[21]:.4f}\n")
            g.write(f"Minimum Forecasting Horizon: {ANS[22]}\n")
            g.write(f"Average Absolute Error: {ANS[23]:.4f}\n")
            g.write(f"Average Error: {ANS[24]:.4f}\n")


            # Calculating and Writing G functions' values
            g.write("\nG Function Values:\n")
            if len(Y) >= 4:  # Ensure there are enough data points
                x1, x2, x3, x4 = Y[-4], Y[-3], Y[-2], Y[-1]  # Last four values from the time series
                for i in range(1, summs_count + 1):
                    g_value = G[i](x1, x2, x3, x4)
                    g.write(f"G{i}({x1}, {x2}, {x3}, {x4}): {g_value}\n")

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
            plot_time_series_adjusted(Y, calculated_ts_values, sn, f'plot_Adjusted_{sn}.png')

            # Call the plotting function with the save path
            plot_time_series(original_data, calculated_data, sn, plot_save_path)
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


# %%
