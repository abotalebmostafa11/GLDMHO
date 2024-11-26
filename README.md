# GLDMHO
Generalized Least Deviation Method High Order
# AbotalebGLDM

**AbotalebGLDM** is a Python library (version 1.0.0) designed for modeling univariate time series using the first-order **Generalized Least Deviation Method (GLDM)**. This method leverages the previous time step value (yₜ₋₁) to forecast the current value (yₜ), providing accurate and efficient predictions for various time series applications.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Running the Model](#running-the-model)
- [Outputs](#outputs)
- [Example](#example)
- [License](#license)

## Features

- **First-Order GLDM**: Utilizes previous time step values for forecasting, making it suitable for univariate time series analysis.
- **Easy Integration**: Simple and intuitive API allows seamless integration into existing projects and workflows.
- **Automated Outputs**: Automatically generates visualizations and evaluation metrics upon model execution.
- **Performance Metrics**: Provides detailed insights into model performance, including evaluation metrics, solution systems, execution time, and memory usage.
- **Lightweight**: Minimal dependencies ensure easy installation and quick setup.
- **Extensible**: Designed to allow future enhancements and integration of higher-order GLDM methods.

## Installation

You can install **AbotalebGLDM** using `pip`, or from the source code.

### Via `pip`

Ensure you have `pip` installed. Then, run:


pip install AbotalebGLDM


From Source
If you prefer to install the package from the source, follow these steps:

Clone the Repository
git clone https://github.com/abotalebmostafa11/GLDMHO 

 ## Data Format
The default input data is expected to be in a file named input.txt. The data structure should follow the format below, which is exemplified using an NDVI dataset:

| **Data**  | **15      1**  |
|-----------|----------------|
| 1         | 0.2950428571   |
| 2         | 0.3935857143   |
| 3         | 0.5285714286   |
| 4         | 0.6218285714   |
| 5         | 0.6637285714   |
| 6         | 0.6701142857   |
| 7         | 0.6759714286   |
| 8         | 0.6935285714   |
| 9         | 0.6907857143   |
| 10        | 0.6777857143   |
| 11        | 0.6159142857   |
| 12        | 0.5291714286   |
| 13        | 0.4574714286   |
| 14        | 0.4132         |
| 15        | 0.3973         |


Explanation:
First Line (15 1):

**15**: Length of the time series data.
**1**: Number of univariate time series.
**Subsequent Lines**: Each line represents a data point in the time series.
**Ensure that your input.txt follows this structure for the library to function correctly.**



## Usage
Running the Model
To utilize the AbotalebGLDM library, follow these simple steps:

### **Prepare Your Data:** Ensure your data is saved in input.txt with the correct format.

### Run the Model:
**import sys**
**from AbotalebGLDM import run**

**run("input.txt")**

**What Happens When You Run the Model**
Model Execution: The GLDM model runs using the first-order method.
**Automated Outputs:**
**Figures:** Visualizations of the time series and forecasting results are saved automatically.
**Output File (output.txt):** Contains model evaluations, metrics, solution systems, time consumption, and memory usage.


## **Outputs**
**After running the model, the following outputs are generated:**
**Figures:** Visual representations of the time series data and forecasting results. These figures are typically saved in formats like .png in the directory where the script is executed.
**output.txt:** A comprehensive report including:
**Model Evaluation Metrics:** Assessing the performance of the GLDM model (e.g., Mean Absolute Error, Root Mean Squared Error).
**Solution System:** Details of the mathematical solution applied by the GLDM.
**Performance Metrics:** Time taken to run the model and memory consumed during execution.


## **Example**
Here's a step-by-step example to demonstrate how to use AbotalebGLDM:

1. Prepare input.txt
Create a file named input.txt in the same directory as your script with the following content:

| **Data**  | **15      1**  |
|-----------|----------------|
| 1         | 0.2950428571   |
| 2         | 0.3935857143   |
| 3         | 0.5285714286   |
| 4         | 0.6218285714   |
| 5         | 0.6637285714   |
| 6         | 0.6701142857   |
| 7         | 0.6759714286   |
| 8         | 0.6935285714   |
| 9         | 0.6907857143   |
| 10        | 0.6777857143   |
| 11        | 0.6159142857   |
| 12        | 0.5291714286   |
| 13        | 0.4574714286   |
| 14        | 0.4132         |
| 15        | 0.3973         |



2. Create and Run the Script
Create a Python script (e.g., run_model.py) with the following content:

import sys
from AbotalebGLDM import run

# Run the GLDM model with the input data
run("input.txt")


3. Review the Outputs
**Figures:** Check the generated visualizations in your directory. These may include plots of the original time series, forecasted values, and residuals.


**output.txt:** Open the file to review model evaluations and performance metrics. This file provides insights into the accuracy and efficiency of the GLDM model applied to your data.



![mainscheme](https://github.com/user-attachments/assets/b0b1b35a-1ac9-4a7d-ae4e-a2978fcd89ad)

![GLDMest](https://github.com/user-attachments/assets/592be036-9b45-4c5e-b526-221fc1367eb6)

![WLDM](https://github.com/user-attachments/assets/94928ea8-bad6-4a80-bb66-ba0aec8c4271)



## License

© 2024 Author: Mostafa Abotaleb


<pre>
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
</pre>



```bash
