# Car-Price-Prediction

This repository contains a machine learning project that predicts car prices based on various features. The project utilizes a dataset of car listings and employs a regression model to estimate the price of a car given its attributes.

## Project Structure

The repository is structured as follows:

- `car_price_app/carmodel.pickle`: This file is a serialized version of the trained machine learning model for car price prediction.
- `car_price_app/encoder.pkl`: This file contains the serialized version of an encoder or preprocessing object used to transform categorical variables into a numerical format suitable for the machine learning model.
- `car_price_app/app.py`: This directory contains a web application for car price prediction. Users can input car details, and the application will return an estimated price based on the trained model.
-  `data/sampled.csv`: This directory contains the dataset used for training and evaluation. It includes car listings with relevant features such as make, model, year, mileage, and more.
- `requirements.txt`: This file lists all the required Python libraries and dependencies to run the project.

## Usage

To use this project, follow these steps:

1. Install the necessary dependencies by running the following command:
```json
    pip install -r requirements.txt 
```
2. **Launching the Web Application:**
   
   Run the `app.py` script from your command line to start the car price prediction web application.

3. **Convenient URL Access**: 
   
   As the application starts, a URL will be displayed in the command line interface. Copy the URL and paste it directly into your browser to access the application. 


## Quick Guide 

To enhance the user experience, several engineering techniques were implemented:

1. When you choose a car brand, the available models for that brand will be updated automatically in the model dropdown menu.

2. Similarly, when you choose a car model, the available body types and fuel types will be updated in the body types and fuel types dropdown menus, respectively.

**NOTE**: The visibility of the car details section is updated when a brand is selected. This means that when you choose a brand, the additional car details section will become visible, including the year, mileage, and other relevant fields.

[**WEBAPP**](https://huggingface.co/spaces/theadedolapo/Car_price_prediction)
