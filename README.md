# Time Series Classification and Forecasting/Regression: Signature vs RNN

This repository contains the code for conducting numerical experiments to compare signature-based and RNN-based architectures for time series classification and forecasting/regression. The experiments aim to evaluate the performance and effectiveness of both approaches in various scenarios.

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
- [Setup](#setup)
- [Usage](#usage)

## Introduction

Time series analysis plays a crucial role in various domains, including finance, healthcare, and climate modeling. This repository explores two different approaches for time series tasks: signature-based methods and recurrent neural network (RNN) architectures.

The signature method leverages the concept of the *signature transform*, which captures the sequential structure of a time series by computing its iterated integrals. On the other hand, RNN-based models, such as LSTM, utilize sequential processing to learn temporal dependencies in the data.

The main objectives of this project are:

1. Implement signature-based and RNN-based models for time series classification and forecasting/regression.
2. Perform numerical experiments to compare the performance of both approaches across different datasets.
3. Provide insights and recommendations based on the experimental results.

## Files

The repository contains the following files:

- `.gitignore`: Specifies the files and directories to be ignored by Git version control.
- `ENV.yml`: A YAML file containing the necessary dependencies and environment setup for running the code.
- `classification.py`: Python script for conducting the experiments related to time series classification. It includes the implementation of signature-based and RNN-based models, as well as evaluation metrics and data preprocessing functions.
- `models.py`: This file contains the implementation of the signature-based and RNN-based models used in the experiments.
- `regression.py`: Python script for performing the experiments related to time series forecasting/regression. It includes the implementation of signature-based and RNN-based models, as well as evaluation metrics and data preprocessing functions.
- `utils.py`: A utility module that provides helper functions for data loading, splitting, and preprocessing.

## Setup

To set up the project environment, please follow these steps:

1. Clone this repository to your local machine using the following command:

        git clone https://github.com/your-username/your-repo.git

2. Create a virtual environment using **conda** (recommended) or **venv** and activate it:
         
- Using **conda**:

  ```
  conda create --name myenv python=3.8
  conda activate myenv
  ```

- Using **venv**:

  ```
  python3 -m venv myenv
  source myenv/bin/activate  # For Linux/Mac
  myenv\Scripts\activate     # For Windows

3. Install the required dependencies by running the following command:
 - Using **conda**:
  ```
  conda env create -f ENV.yml
  ```
 - Using **venv**:

  ```
  pip install -r ENV.yml
  ```
 
This will install all the necessary packages and dependencies needed to run the code.

## Usage

To use the code in this repository, follow these guidelines:

1. Ensure that you have set up the project environment as described in the [Setup](#setup) section.

2. Modify the data loading and preprocessing functions in the respective `classification.py` and `regression.py` scripts to suit your specific dataset requirements.

3. Run the desired script based on your task:

- For time series classification experiments, execute the following command:

  ```
  python classification.py
  ```

- For time series forecasting/regression experiments, execute the following command:

  ```
  python regression.py
  ```

These scripts will run the experiments using the specified models, evaluate their performance, and provide relevant output and visualizations.

4. Analyze the experimental results and compare the performance of the signature
 
