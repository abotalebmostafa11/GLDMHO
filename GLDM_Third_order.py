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

# Function pointers are represented as elements in a list
G = [None, None, None, None, None, None]


def G1(x1, x2, x3):
    return x1

def G2(x1, x2, x3):
    return x2

def G3(x1, x2, x3):
    return x3

def G4(x1, x2, x3):
    return x1 * x1

def G5(x1, x2, x3):
    return x2 * x2

def G6(x1, x2, x3):
    return x3 * x3

def G7(x1, x2, x3):
    return x1 * x2

def G8(x1, x2, x3):
    return x1 * x3

def G9(x1, x2, x3):
    return x2 * x3

def GForming():
    global summs_count  # Ensure you use the global variable

    # Initialize G as a list with enough elements
    G.extend([None] * (summs_count + 1 - len(G)))

    # Assign values to G functions
    G[1] = G1
    G[2] = G2
    G[3] = G3
    G[4] = G4
    G[5] = G5
    G[6] = G6
    G[7] = G7
    G[8] = G8
    G[9] = G9

# Update summs_count for the new G functions
#summs_count = 9  # Updated count

def SSTForming(_Y):
    # Creating the _SST matrix. In Python, we usually use lists or NumPy arrays for matrices.
    _SST = [[0.0 for _ in range(summs_count * 2 + 2)] for _ in range(summs_count + 1)]

    # Iterating over the matrix to perform calculations
    for i in range(1, summs_count + 1):
        for j in range(1, summs_count + 1):
            for k in range(4, impl_len + 1):  # Start from 4 to include the third previous value
                _SST[i][j] += G[i](_Y[k - 3], _Y[k - 2], _Y[k - 1]) * G[j](_Y[k - 3], _Y[k - 2], _Y[k - 1])

        # Setting specific matrix elements to 0.0 and 1.0
        for j in range(1, summs_count + 1):
            _SST[i][summs_count + j] = 0.0
        _SST[i][summs_count + i] = 1.0

    # Printing the matrix
    print('\nMatrix SST\n')
    for i in range(1, summs_count + 1):
        print('\n', i, '\t', end='')
        for j in range(1, summs_count * 2 + 1):
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

        # Printing the matrix for each step (optional)
        print('\nMatrix SST after iteration', iter_first)
        for i in range(1, nn + 1):
            print('\n', i, '\t', end='')
            for j in range(1, nn + nn + 1):
                print(_SST[i][j], '\t', end='')



def P1Forming(_Y, _SST):
    _P1 = [[0.0 for _ in range(summs_count + 1)] for _ in range(impl_len + 2)]

    for t in range(4, impl_len + 1):  # Starting from 4 to account for third-order
        for j in range(1, summs_count + 1):
            _P1[t][j] = 0.0

            for k in range(1, summs_count + 1):
                _P1[t][j] += G[k](_Y[t - 3], _Y[t - 2], _Y[t - 1]) * _SST[k][summs_count + j]

    # Printing the matrix (optional, for verification)
    print('\nMatrix P1[4:m][1:n]\n')
    for iter_first in range(4, impl_len + 1):
        print('\n', iter_first, '\t', end='')
        for iter_second in range(1, summs_count + 1):
            print(_P1[iter_first][iter_second], '\t', end='')

    return _P1


def PForming(_Y, _P1):
    # Initialize the _P matrix
    _P = [[0.0 for _ in range(impl_len + 2)] for _ in range(impl_len + 2)]

    for iter_first in range(4, impl_len + 1):  # Start from 4 for third-order
        for iter_second in range(4, impl_len + 1):
            _P[iter_first][iter_second] = 0.0

            for iter_third in range(1, summs_count + 1):
                _P[iter_first][iter_second] -= G[iter_third](_Y[iter_second - 3], _Y[iter_second - 2], _Y[iter_second - 1]) * _P1[iter_first][iter_third]

            if iter_first == iter_second:
                _P[iter_first][iter_first] += 1.0

    # Printing the matrix (optional, for verification)
    print('\nMatrix P[4:m][4:m]\n')
    for iter_first in range(4, impl_len + 1):
        print('\n', iter_first, '\t', end='')
        for iter_third in range(4, impl_len + 1):
            print(_P[iter_first][iter_third], '\t', end='')

    return _P


def PrGradForming(_Y, _P):
    # Initialize the _Prgrad and _grad arrays
    _Prgrad = [0.0 for _ in range(impl_len + 2)]
    _grad = [0.0 for _ in range(impl_len + 2)]

    # Copying _Y values to _grad
    for i in range(1, impl_len + 2):
        _grad[i] = _Y[i]

    for iter_first in range(4, impl_len + 1):  # Start from 4 for third-order
        _Prgrad[iter_first] = 0.0
        for iter_second in range(4, impl_len + 1):
            _Prgrad[iter_first] += _P[iter_first][iter_second] * _grad[iter_second]

    # Printing the results (optional for debugging)
    print('\ni   grad[i]   Prgrad[i]    p[i]  \n', end='')
    for iter_first in range(4, impl_len + 1):
        print(f'\n{iter_first}\t{_grad[iter_first]}\t{_Prgrad[iter_first]}\t', end='')

    return _Prgrad


def DualWLDMSolution(_w, _p, _Prgrad):
    Al = LARGE
    Alc = 0

    for iter_first in range(4, impl_len + 1):  # Start from 4 for third-order
        _w[iter_first] = 0

    iter_first = 0
    while iter_first < impl_len - summs_count - 2:
        Al = LARGE
        for iter_second in range(4, impl_len + 1):  # Adjusted for third-order
            if abs(_w[iter_second]) == _p[iter_second]:
                continue
            else:
                if _Prgrad[iter_second] > 0:
                    Alc = (_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]
                elif _Prgrad[iter_second] < 0:
                    Alc = (-_p[iter_second] - _w[iter_second]) / _Prgrad[iter_second]

                if Alc < Al:
                    Al = Alc

        for iter_second in range(4, impl_len + 1):  # Adjusted for third-order
            if abs(_w[iter_second]) != _p[iter_second]:
                _w[iter_second] += Al * _Prgrad[iter_second]
                if abs(_w[iter_second]) == _p[iter_second]:
                    iter_first += 1


def PrimalWLDMSolution(_Y, _SST, _w, _p, _a, _z):
    lc_r = [0 for _ in range(summs_count + 1)]  # Ordinal numbers of the basic equations
    lc_ri = 0  # The amount of basic equations of the primal problem

    for iter_first in range(4, impl_len + 1):  # Adjust for third-order
        if abs(_w[iter_first]) != _p[iter_first]:
            lc_ri += 1
            lc_r[lc_ri] = iter_first

    for iter_first in range(1, lc_ri + 1):
        for iter_second in range(1, lc_ri + 1):
            _SST[iter_first][iter_second] = G[iter_second](_Y[lc_r[iter_first] - 3], _Y[lc_r[iter_first] - 2], _Y[lc_r[iter_first] - 1])

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

        Z = lc_z[1] = lc_z[2] = lc_z[3] = 0.0
        for i in range(4, impl_len + 1):  # Start from 4 for third-order
            lc_z[i] = _Y[i]
            for j in range(1, summs_count + 1):
                lc_z[i] -= lc_a[j] * G[j](_Y[i - 3], _Y[i - 2], _Y[i - 1])
            Z += abs(lc_z[i])

        d = max([abs(lc_a[i] - lc_a1[i]) for i in range(1, summs_count + 1)])
        if d < 0.5:  # some_tolerance_value: Define as per your requirements
            break

    # Constructing the solution
    Sol = Solution()
    Sol.a = lc_a
    Sol.z = lc_z
    Sol.Z = Z

    return Sol



def ForecastingEst(Y, Sol):
    PY = [[0.0 for _ in range(len(Y))] for _ in range(len(Y))]
    FH = [0 for _ in range(len(Y))]
    e = Errors()

    # Iterate through each starting point in the series
    for St in range(len(Y) - 3):  # Adjust for third-order initial conditions
        PY[St][0], PY[St][1], PY[St][2] = Y[St:St+3]
        t = 3

        while True:
            # Break if forecasting goes beyond the bounds of Y
            if St + t >= len(Y):
                break

            # Calculate forecast using third-order model
            py = sum(Sol.a[j] * G[j](PY[St][t - 1], PY[St][t - 2], PY[St][t - 3]) for j in range(1, len(Sol.a)))
            PY[St][t] = py

            # Increment t for next forecasting step
            t += 1

            # Break if the forecast error exceeds the tolerance
            if abs(PY[St][t - 1] - Y[St + t - 1]) > Sol.Z:
                break

        # Record the forecasting horizon
        FH[St] = t - 1

    # Calculate minimum forecasting horizon, ensuring it's greater than 3
    e.minFH = min([h for h in FH if h > 3])

    # Calculate errors for the forecasting horizon, ensuring there's no division by zero
    if e.minFH > 3:
        e.E, e.D = 0, 0
        for St in range(len(Y) - 3):
            for t in range(4, e.minFH + 1):
                if St + t < len(Y):
                    e.D += abs(Y[St + t] - PY[St][t])
                    e.E += (Y[St + t] - PY[St][t])
        total_points = sum(FH) - 3 * len([h for h in FH if h > 3])
        if total_points > 0:
            e.E /= total_points
            e.D /= total_points
    else:
        # Handle case where minFH is not greater than 3 or division by zero would occur
        e.E, e.D = float('inf'), float('inf')  # Indicate an error or undefined state

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
    Calculate time series values using the coefficients in Solution for a third-order model.

    Parameters:
    Y (list): The initial values of the time series.
    Sol (Solution): The solution object containing the coefficients.
    length (int): The number of time series values to calculate.

    Returns:
    list: Calculated time series values.
    """
    calculated_values = [0.0 for _ in range(length)]

    # Assuming the first three values of Y are initial values
    calculated_values[0] = Y[0]
    calculated_values[1] = Y[1]
    calculated_values[2] = Y[2]

    # Calculate the rest of the values based on the coefficients
    for t in range(3, length):
        value = Sol.a[0]  # This could be an intercept if your model has one
        for i in range(1, len(Sol.a)):
            value += Sol.a[i] * G[i](Y[t - 1], Y[t - 2], Y[t - 3])
        calculated_values[t] = value

    return calculated_values

def plot_time_series(original_data, calculated_data, series_number, save_path=None):
    plt.figure(figsize=(10, 6))

    # Ensure the x_axis_range matches the length of your data.
    # The range should start from 0 (or 1 if you want to start from the first data point) 
    # and extend to the length of your data.
    x_axis_range = range(len(original_data))

    # Plot the original and calculated data
    plt.plot(x_axis_range[1:-1], original_data[1:-1], label='Original', color='blue', linewidth=2)
    plt.plot(x_axis_range[1:-1], calculated_data[1:-1], label='GLDM Model', color='red', linestyle='dotted', linewidth=2)

    plt.title(f'Time Series: {case_name}: Original vs GLDM Model', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)

    plt.show()
    plt.close()





def plot_time_series_adjusted(original_data, calculated_data, series_number, save_path):
    plt.figure(figsize=(10, 6))

    # Ensure the first four values of calculated_data match the original_data
    if len(original_data) >= 4 and len(calculated_data) >= 4:
        for i in range(4):  # Update this loop to copy the first four values
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
    plt.title(f'Time Series: {case_name}: Original vs GLDM Model', fontsize=16)
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
        summs_count = 9  # Updated for third-order (total number of G functions)
        print(f"Length: {m}\nTime series: {ts}\n")
        
        # Reading time series data
        setnum = 0
        RY = [[] for _ in range(ts)]  # Create the ts arrays for time series
        for i in range(ts):
            RY[i] = [0.0] * (m + 2)
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
        GL_RY = RY
        # End Reading time series data

        # Writing results to a file
        g.write(f"Number of time series: {ts}\n")
        g.write(f"Length of time series: {impl_len}\n")

        # Processing each time series
        for sn in range(ts):
            Y = [0.0] * (m + 2)
            for j in range(1, m + 1):
                Y[j] = RY[sn][j]

            GL_Y = GL_RY[sn]

            # Solution
            GForming()
            print("GForming() OK\n")
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
            start_index = 4 # Change to 2 if you need to skip the first two values for some reason

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
            g.write(f"Error Matrix start from Fourth point to the end of dataset\n")
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
                g.write(f"{orig:<20.4f}{calc:<20.4f}{error:<20.4f}\n")  # Adjusted for floating point display

            # Write model coefficients and errors
            # Write model coefficients and errors
            ANS = [0] * 15
            ANS[0] = sn  # Time Series Number
            ANS[1] = 0   # Placeholder for future use or additional data
            for i in range(summs_count):
                ANS[i + 2] = Sol.a[i + 1]  # Model Coefficients
            ANS[11] = Sol.Z  # Sum of Absolute Differences between Model and Actual Data
            e = ForecastingEst(GL_Y, Sol)  # Forecasting Errors
            print("ForecastingEST OK\n")
            print(e.minFH,"\n", end='')
            ANS[12] = e.minFH  # Minimum Forecasting Horizon
            ANS[13] = e.D     # Average Absolute Error
            ANS[14] = e.E    # Average Error

            # Writing the results with descriptive labels
            g.write(f"Time Series Number: {ANS[0]}\n")
            g.write("Model Coefficients:\n")
            for i in range(2, 11):
                g.write(f"Coefficient a{i-1}: {ANS[i]}\n")
            g.write(f"Sum of Absolute Differences: {ANS[11]}\n")
            g.write(f"Minimum Forecasting Horizon: {ANS[12]}\n")
            g.write(f"Average Absolute Error: {ANS[13]}\n")
            g.write(f"Average Error: {ANS[14]}\n")
            # Calculating and Writing G functions' values
            g.write("\nG Function Values:\n")
            if len(Y) >= 4:  # Ensure there are enough data points
                x1, x2, x3 = Y[1], Y[2], Y[3]  # Example arguments from the time series
                for i in range(1, summs_count + 1):
                    g_value = G[i](x1, x2, x3)
                    g.write(f"G{i}({x1}, {x2}, {x3}): {g_value}\n")

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

 

            # Optional: Plotting
            plot_time_series(Y, calculated_ts_values, sn, f'plot_series_{sn}.png')
            plot_time_series_adjusted(Y, calculated_ts_values, sn, f'plot_Adjusted_{sn}.png')




        # Measure and write performance metrics
        end_time = time.time()
        final_memory_use = process.memory_info().rss / (1024 * 1024)
        total_time = end_time - start_time
        total_memory_used = final_memory_use - initial_memory_use
        g.write("\nPerformance Metrics:\n")
        g.write(f"Total Execution Time: {total_time:.2f} seconds\n")
        g.write(f"Total Additional Memory Used: {total_memory_used:.2f} MB\n")




    # Wait for user input before closing
    input("\nPress any key: ")

if __name__ == "__main__":
    main()