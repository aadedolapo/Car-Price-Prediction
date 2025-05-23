import pickle
import gradio as gr
import pandas as pd

from beartype import beartype
from car_predictor.predict_price import Predictor


df = pd.read_csv("data/sampled.csv")


@beartype
def predict_car_price(
    brand: str,
    model: str,
    reg_year: int,
    mileage: str,
    condition: str,
    fuel: str,
    body: str,
    colour: str
) -> str:

    # Create an instance of the Predictor class
    user_input = Predictor(brand, model, reg_year, mileage,
                           condition, fuel, body, colour)

    # Call the predict_car method and return the result
    predicted_price = user_input.predict_price()
    return predicted_price


@beartype
def search_car_model(search_term: str) -> list:
    result_df = df[df['standard_make'].str.contains(search_term)]
    return result_df['standard_model'].drop_duplicates().tolist()


@beartype
def search_body(model: str) -> list:
    result_df = df[df['standard_model'] == model]
    return result_df['body_type'].drop_duplicates().tolist()


@beartype
def search_fuel(model: str) -> list:
    result_df = df[df['standard_model'] == model]
    return result_df['fuel_type'].drop_duplicates().tolist()


with gr.Blocks(gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # CAR PRICE PREDICTION

    ### QUICK GUIDE
    To optimize the user experience, several engineering techniques were implemented:
    1. When you choose a car brand, the available models for that brand will be updated automatically in the model dropdown menu. 
    2. Similarly, when you choose a car model, the available body types and fuel types will be updated in the body types and fuel types dropdown menu respectively.
    <br />
    <br />
    **NOTE**: the visibility of the car details section is updated when a brand is selected. This means that when you choose a brand, the additional car details section will become visible, year of registration, mileage, condition of the car(new used), and  specific details about the car. 
    
    """
    )
    colour = ['BEIGE', 'BLACK', 'BLUE', 'BRONZE', 'BROWN', 'GREEN', 'GREY', 'MULTICOLOUR',
              'ORANGE', 'PURPLE', 'RED', 'SILVER', 'WHITE', 'YELLOW', 'OTHER COLOUR']
    brands = df['standard_make'].drop_duplicates().values.tolist()
    models = df['standard_model'].drop_duplicates().values.tolist()
    car_brand = gr.Dropdown(label="Brand", choices=brands)
    car_model = gr.Dropdown(label="Model", choices=[])
    body = gr.Dropdown(label='Body Type', choices=[])
    fuel = gr.Dropdown(label='Fuel Type', choices=[])

    with gr.Column(visible=False) as details_col:
        year = gr.Slider(1990, 2020, step=1, label="Year of Registration")
        mileage = gr.Textbox(label="Mileage", placeholder="Input mileage...")
        condition = gr.Radio(["New", "Used"], label="Vehicle Condition")
        colour = gr.Dropdown(colour, label="Colour")
        generate_btn = gr.Button("Predict Price")
        gr.Markdown("Predicted Price:")
        output = gr.Text(label="Predict Price")

    brand_models = {brand: search_car_model(brand) for brand in brands}
    model_body = {model: search_body(model) for model in models}
    model_fuel = {model: search_fuel(model) for model in models}

    def filter_models(car_brand):
        return gr.update(
            choices=brand_models[car_brand], value=brand_models[car_brand][0]
        ), gr.update(visible=True)

    car_brand.change(filter_models, car_brand, [car_model, details_col])

    def filter_body(car_model):
        return gr.update(
            choices=model_body[car_model], value=model_body[car_model][0]
        ), gr.update(visible=True)

    car_model.change(filter_body, car_model, [body, details_col])

    def filter_fuel(car_model):
        return gr.update(
            choices=model_fuel[car_model], value=model_fuel[car_model][0]
        ), gr.update(visible=True)

    car_model.change(filter_fuel, car_model, [fuel, details_col])

    generate_btn.click(fn=predict_car_price, inputs=[
                       car_brand, car_model, year, mileage, condition, fuel, body, colour], outputs=output)


if __name__ == "__main__":
    demo.launch()
