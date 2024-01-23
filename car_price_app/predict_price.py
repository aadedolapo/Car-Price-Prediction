import pickle
import gradio as gr
import pandas as pd

from beartype import beartype


class Predictor:
    """Predicts the price of a car

    This class provides a convenient way to predict the price of a car given features 
    like brand, model, year of first registration, mileage, vehicle condition, 
    fuel type, and body type.
    """

    df = pd.read_csv("data/sampled.csv")

    @beartype
    def __init__(
        self,
        brand: str,
        model: str,
        reg_year: int,
        mileage: str,
        condition: str,
        fuel: str,
        body: str,
        colour: str
    ) -> None:
        """Initializes the Predictor class.

            Args:
                brand (str): The parent name of the car.

                model (str): The model of the car.

                reg_year (int): The year the car was first registered.

                mileage (int): The mileage of the car in miles per hour.

                condition (str): The condition of the car. Either used or new.

                fuel (str): The fuel type of the car.

                body (str): The body type of the car

                colour (str): The colour of the car.

            Returns:
                float: The predicted price of the car.
            """
        self.brand = brand
        self.model = model
        self.reg_year = reg_year
        self.mileage = int(mileage)
        self.condition = condition
        self.fuel = fuel
        self.body = body
        self.colour = colour

    @beartype
    def predict_price(self) -> str:
        input_features = pd.DataFrame({
            'mileage': [self.mileage],
            'standard_colour': [self.colour],
            'standard_make': [self.brand],
            'standard_model': [self.model],
            'vehicle_condition': [self.condition],
            'year_of_registration': [self.reg_year],
            'body_type': [self.body],
            'fuel_type': [self.fuel],
        })
        enc = pickle.load(open("models/encoder.pkl", "rb"))
        input_features = enc.transform(input_features)
        model = pickle.load(open("models/carmodel.pickle", "rb"))

        # Make predictions on the input features
        predicted_price = model.predict(input_features)

        return "£{}".format(round(predicted_price[0], 2))


if __name__ == "__main__":
    Predictor()
