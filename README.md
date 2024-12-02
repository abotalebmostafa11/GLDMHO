# GLDMHO
Generalized Least Deviation Method High Order
# abotaleb1

**abotaleb1** is a Python package (version 1.0.0) designed for modeling univariate time series using the first to fifth-order **Generalized Least Deviation Method (GLDM)**. This method leverages previous time step values ($y_{t-1}$, $y_{t-2}$, $y_{t-3}$, $y_{t-4}$, $y_{t-5}$) to forecast the current value ($y_t$), providing accurate and efficient predictions for various time series applications.


### Model Details and Coefficients

In the Generalized Least Deviation Method (GLDM), the number of coefficients and the lagged variables used increase with the order of the method:

- **First Order**: Uses one lagged variable to forecast $y_t$.
  - **Lagged Variables**: $y_{t-1}$
  - **Coefficients**: 2 coefficients
    - $a_1$, $a_2$

- **Second Order**: Uses two lagged variables to forecast $y_t$.
  - **Lagged Variables**: $y_{t-1}$, $y_{t-2}$
  - **Coefficients**: 5 coefficients
    - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$

- **Third Order**: Uses three lagged variables to forecast $y_t$.
  - **Lagged Variables**: $y_{t-1}$, $y_{t-2}$, $y_{t-3}$
  - **Coefficients**: 9 coefficients
    - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$, $a_6$, $a_7$, $a_8$, $a_9$

- **Fourth Order**: Uses four lagged variables to forecast $y_t$.
  - **Lagged Variables**: $y_{t-1}$, $y_{t-2}$, $y_{t-3}$, $y_{t-4}$
  - **Coefficients**: 14 coefficients
    - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$, $a_6$, $a_7$, $a_8$, $a_9$, $a_{10}$, $a_{11}$, $a_{12}$, $a_{13}$, $a_{14}$

- **Fifth Order**: Uses five lagged variables to forecast $y_t$.
  - **Lagged Variables**: $y_{t-1}$, $y_{t-2}$, $y_{t-3}$, $y_{t-4}$, $y_{t-5}$
  - **Coefficients**: 20 coefficients
    - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$, $a_6$, $a_7$, $a_8$, $a_9$, $a_{10}$, $a_{11}$, $a_{12}$, $a_{13}$, $a_{14}$, $a_{15}$, $a_{16}$, $a_{17}$, $a_{18}$, $a_{19}$, $a_{20}$

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Running the Model](#running-the-model)
- [Outputs](#outputs)
- [Example](#example)
- [Example](#ALGEORITHMS SCHEMA)
- [License](#license)

## Features
- **First-Order GLDM**: Utilizes previous time step values for forecasting, making it suitable for univariate time series analysis.
- **Easy Integration**: Simple and intuitive API allows seamless integration into existing projects and workflows.
- **Automated Outputs**: Automatically generates visualizations and evaluation metrics upon model execution.
- **Performance Metrics**: Provides detailed insights into model performance, including evaluation metrics, solution systems, execution time, and memory usage.
- **Lightweight**: Minimal dependencies ensure easy installation and quick setup.
- **Extensible**: Designed to allow future enhancements and integration of higher-order GLDM methods.
## Installation
You can install **gldmabotaleb** using `pip`, or from the source code.
### Via `pip`
Ensure you have `pip` installed. Then, run:
pip install gldmabotaleb
From Source
If you prefer to install the package from the source, follow these steps:
Clone the Repository
git clone https://github.com/abotalebmostafa11/GLDMHO 
gitverse https://gitverse.ru/mostafa/GLDM?tab=readme 

 ## Data Format
## Input File Format
The `input.txt` file should be formatted as follows:

| **Line** | **Content** |                    **Description**                                 |
|----------|-------------|--------------------------------------------------------------------|
| 1        | `:`         | Separator indicating the start of data sections                    |
| 2        | `m ts`      | - `m`: Length of the time series<br `ts`                           |
| 3        | `yt`        | First data point of the first time series                          |
| 4        | `yt_1`      | Second data point of the first time series                         |
| 5        | `yt_2`      | Third data point of the first time series                          |
| ...      | `...`       | ...                                                                |
| `m*ts + 2` | `yt_m`    | `m`-th data point of the `ts`-th time series                       |


The default input data is expected to be in a file named input.txt. The data structure should follow the format below, which is exemplified using an NDVI dataset:

|**Data:**  **15      1**    |
|----------------------------|
|0.2950428571                |
|0.3935857143                |
|0.5285714286                |
|0.6218285714                |
|0.6637285714                |
|0.6701142857                |
|0.6759714286                |
|0.6935285714                |
|0.6907857143                |
|0.6777857143                |
|0.6159142857                |
|0.5291714286                |
|0.4574714286                |
|0.4132                      |
|0.3973                      |

**Explanation:**
**First Line (15 1):**
**15**: Length of the time series data.
**1**: Number of univariate time series (in this example we have only one time series).
**Subsequent Lines**: Each line represents a data point in the time series.
**Ensure that your input.txt follows this structure for the library to function correctly.**
## Usage
Running the Model
To utilize the gldmabotaleb library, follow these simple steps:
### **Prepare Your Data:** Ensure your data is saved in input.txt with the correct format.
### Run the Model:
**import sys**
**from gldmabotaleb import run**
**run("input.txt")**
**What Happens When You Run the Model**
Model Execution: The GLDM model runs using the first-order method.
**Automated Outputs:**
**Figures:** Visualizations of the time series and forecasting results are saved automatically.
**Output File (output.txt):** Contains model evaluations, Model coefficients ($a_1$, $a_2$), metrics, solution systems, time consumption, and memory usage.

**Generalized Least Deviation Method (GLDM)**

The Generalized Least Deviation Method (GLDM) is used for modeling univariate time series. In GLDM, the number of coefficients increases with the order of the method:

- **First Order**: 2 coefficients
  - $a_1$, $a_2$

- **Second Order**: 5 coefficients
  - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$

- **Third Order**: 9 coefficients
  - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$, $a_6$, $a_7$, $a_8$, $a_9$

- **Fourth Order**: 14 coefficients
  - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$, $a_6$, $a_7$, $a_8$, $a_9$, $a_{10}$, $a_{11}$, $a_{12}$, $a_{13}$, $a_{14}$

- **Fifth Order**: 20 coefficients
  - $a_1$, $a_2$, $a_3$, $a_4$, $a_5$, $a_6$, $a_7$, $a_8$, $a_9$, $a_{10}$, $a_{11}$, $a_{12}$, $a_{13}$, $a_{14}$, $a_{15}$, $a_{16}$, $a_{17}$, $a_{18}$, $a_{19}$, $a_{20}$



## **Outputs**
**After running the model, the following outputs are generated:**
**Figures:** Visual representations of the time series data and forecasting results. These figures are typically saved in formats like .png in the directory where the script is executed.
**output.txt:** A comprehensive report including:
**Model Evaluation Metrics:** Assessing the performance of the GLDM model (e.g., Mean Absolute Error, Root Mean Squared Error).
**Solution System:** Details of the mathematical solution applied by the GLDM.
**Performance Metrics:** Time taken to run the model and memory consumed during execution.
## **Example**
Here's a step-by-step example to demonstrate how to use gldmabotaleb:
1. Prepare input.txt
Create a file named input.txt in the same directory as your script with the following content:

|**Data:**  **15      1**    |
|----------------------------|
|0.2950428571                |
|0.3935857143                |
|0.5285714286                |
|0.6218285714                |
|0.6637285714                |
|0.6701142857                |
|0.6759714286                |
|0.6935285714                |
|0.6907857143                |
|0.6777857143                |
|0.6159142857                |
|0.5291714286                |
|0.4574714286                |
|0.4132                      |
|0.3973                      |

2. Create and Run the Script
Create a Python script (e.g., run_model.py) with the following content:
import sys
from gldmabotaleb import run
# Run the GLDM model with the input data
run("input.txt")
3. Review the Outputs
**Figures:** Check the generated visualizations in your directory. These may include plots of the original time series, forecasted values, and residuals.
**output.txt:** Open the file to review model evaluations and performance metrics. This file provides insights into the accuracy and efficiency of the GLDM model applied to your data.
## **ALGEORITHMS SCHEMA:** 
![GLDMest](https://github.com/user-attachments/assets/5f5ec1ba-c610-46eb-adbb-ee7d42639948)


# GLDM Algorithm

## Overview

The Generalized Least Deviation Method (GLDM) is an iterative optimization algorithm designed for modeling and forecasting time series data. It minimizes deviations between observed and predicted values using weighted least deviations. This method efficiently handles both linear and nonlinear patterns in data.

---

## Steps of the Algorithm

### 1. Initialization

- **Input**:
  - Time series data \( S = \{ S_t \in \mathbb{R}^n \}_{t \in T} \).
  - Gradient function \( \nabla \mathcal{L} \).
  - Historical observations \( \{ y_t \}_{t=1-m}^T \).
- **Weights**:
  - Initialize \( p_t = 1 \) for all \( t \in \{1, 2, \ldots, T\} \).

---

### 2. Initial Estimation

- Apply the Weighted Least Deviation Method (WLDM) to estimate the initial coefficients \( A^{(1)} \) and auxiliary variables \( z^{(1)} \):

  $$
  ( A^{(1)}, z^{(1)} ) \leftarrow \text{WLDM}\big( S, \nabla \mathcal{L}, \{ p_t \}_{t=1}^T,\  \{ y_t \}_{t=1-m}^T \big)
  $$

---

### 3. Update Weights

- For every \( t \), update the weights based on the current auxiliary variable \( z_t^{(1)} \):

  $$
  p_t \leftarrow \frac{1}{1 + \big( z_t^{(1)} \big)^2}
  $$

---

### 4. Iterative Optimization

- Start with \( k = 2 \) and repeat the following steps:

  1. Reapply the WLDM to obtain updated estimates \( A^{(k)} \) and \( z^{(k)} \):

     $$
     ( A^{(k)}, z^{(k)} ) \leftarrow \text{WLDM}\big( S, \nabla \mathcal{L}, \{ p_t \}_{t=1}^T,\  \{ y_t \}_{t=1-m}^T \big)
     $$

  2. Update the weights:

     $$
     p_t^{(k)} \leftarrow \frac{1}{1 + \big( z_t^{(k)} \big)^2}, \quad \forall t
     $$

  3. **Check for convergence**:

     - If \( A^{(k)} \neq A^{(k-1)} \), increment \( k \) and repeat.
     - Otherwise, proceed to the next step.

---

### 5. Convergence Check

- The algorithm terminates when \( A^{(k)} \approx A^{(k-1)} \), ensuring the parameters have converged.

---

### 6. Output

- The final coefficients \( A^{(k)} \) and auxiliary variables \( z^{(k)} \) are returned.

---

## Key Features

- **Dynamic Weights**: The algorithm adjusts weights \( p_t \) to reduce the impact of outliers.
- **Iterative Refinement**: Repeated adjustments improve accuracy and robustness.
- **Nonlinear Data Handling**: Capable of modeling complex, nonlinear patterns.

---

## Applications

- Time series forecasting with complex dynamics.
- Scenarios where robustness against outliers is essential.
- High-accuracy modeling and forecasting tasks.

---

# Numerical Example of Generalized Least Deviation Method (GLDM)

## Problem Statement

Given a time series dataset:

$$
S = \{ y_1, y_2, y_3, y_4, y_5 \} = \{ 5.0,\ 4.8,\ 5.2,\ 5.1,\ 4.9 \}
$$

We aim to forecast the next value \( y_6 \) using the Generalized Least Deviation Method.

---

## Steps

### 1. Initialization

- Initialize weights:

  $$
  p_t = 1, \quad \forall t \in \{1,\ 2,\ 3,\ 4,\ 5\}
  $$

- Define the gradient function \( \nabla \mathcal{L} \) (based on the objective function).

---

### 2. First Estimation

Using the initial weights \( p_t = 1 \), apply WLDM to estimate the coefficients \( A^{(1)} \) and auxiliary variables \( z^{(1)} \).

- **Objective function**:

  $$
  \sum_{t=1}^5 p_t \cdot \arctan\big( \left| \hat{y}_t - y_t \right| \big) \to \min
  $$

- **Initial coefficients** (after solving):

  $$
  A^{(1)} = \{ a_1 = 0.5,\ a_2 = 0.3 \}
  $$

- **Auxiliary variables**:

  $$
  z^{(1)} = \{ z_1^{(1)},\ z_2^{(1)},\ z_3^{(1)},\ z_4^{(1)},\ z_5^{(1)} \} = \{ 0.10,\ 0.15,\ 0.20,\ 0.10,\ 0.12 \}
  $$

---

### 3. Update Weights

Update weights using:

$$
p_t \leftarrow \frac{1}{1 + \big( z_t^{(1)} \big)^2}
$$

- **Compute updated weights**:

  $$
  \begin{aligned}
  p_1 &= \frac{1}{1 + (0.10)^2} = 0.9901, \\
  p_2 &= \frac{1}{1 + (0.15)^2} = 0.9775, \\
  p_3 &= \frac{1}{1 + (0.20)^2} = 0.9615, \\
  p_4 &= \frac{1}{1 + (0.10)^2} = 0.9901, \\
  p_5 &= \frac{1}{1 + (0.12)^2} = 0.9852.
  \end{aligned}
  $$

---

### 4. Iterative Optimization

For \( k = 2 \), reapply WLDM using the updated weights \( \{ p_t \} \).

- **Solve for new coefficients**:

  $$
  A^{(2)} = \{ a_1 = 0.52,\ a_2 = 0.28 \}
  $$

- **Update auxiliary variables**:

  $$
  z^{(2)} = \{ z_1^{(2)},\ z_2^{(2)},\ z_3^{(2)},\ z_4^{(2)},\ z_5^{(2)} \} = \{ 0.08,\ 0.13,\ 0.18,\ 0.09,\ 0.10 \}
  $$

**Check convergence**:

- Compare \( A^{(2)} \) and \( A^{(1)} \):

  $$
  A^{(2)} \neq A^{(1)}
  $$

- Increment \( k \) and repeat.

---

### 5. Convergence

After \( k = 5 \), coefficients stabilize:

$$
A^{(5)} = \{ a_1 = 0.55,\ a_2 = 0.27 \}
$$

**Auxiliary variables**:

$$
z^{(5)} = \{ z_1^{(5)},\ z_2^{(5)},\ z_3^{(5)},\ z_4^{(5)},\ z_5^{(5)} \} = \{ 0.05,\ 0.08,\ 0.12,\ 0.06,\ 0.07 \}
$$

---

### 6. Forecasting

Using the final coefficients \( A^{(5)} \), forecast \( y_6 \):

$$
y_6 = a_1 \cdot y_5 + a_2 \cdot y_4
$$

**Substitute values**:

$$
\begin{aligned}
y_6 &= 0.55 \times 4.9 + 0.27 \times 5.1 \\
    &= 2.695 + 1.377 \\
    &= 5.072
\end{aligned}
$$

---

## Final Output

- **Predicted Value**: \( y_6 = 5.072 \)
- **Final Coefficients**: \( A^{(5)} = \{ a_1 = 0.55,\ a_2 = 0.27 \} \)
- **Auxiliary Variables**: \( z^{(5)} = \{ 0.05,\ 0.08,\ 0.12,\ 0.06,\ 0.07 \} \)





![WLDM](https://github.com/user-attachments/assets/b6d66552-70be-4d27-ac5b-479eefe36991)


![GLDM](https://github.com/user-attachments/assets/235b6e1d-c595-426b-ac75-4fd661d76ce1)




## License
© 2024 Author: Mostafa Abotaleb
<pre>
MIT License
</pre>
