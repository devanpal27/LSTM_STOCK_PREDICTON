# Stock Prediction Web Application

This Flask-based web application allows users to upload stock market data in CSV format and receive predictions for the next day's stock price. The application processes the uploaded data, runs a stock prediction model, and displays the results, including a graph of the predictions.

## Features
- Upload CSV files containing stock data.
- Predict the next day's stock price using a pre-trained model.
- View results and a graph of the stock predictions.

## Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.7 or later
- Flask
- Required Python libraries (see `requirements.txt`)

- Install the required dependencies:
pip install -r requirements.txt

## Installation
1. Clone this repository:
   
   git clone https://github.com/your-username/stock-prediction-app.git
   cd stock-prediction-app
Usage
Run the Flask application:
python app.py

Open a web browser and navigate to:
http://127.0.0.1:5000

Upload a CSV file containing stock market data.

View the predicted stock price for the next day along with a graphical representation.

File Structure
app.py: Main Flask application.
templates/index.html: Upload form.
templates/result.html: Result page displaying predictions.
uploads/: Directory for storing uploaded files.
Model/StockPrediction.py: Stock prediction logic.
Key Functionality
File Upload: Ensures only CSV files are accepted.
Prediction Logic: Utilizes the run_stock_prediction function for analyzing stock data.
Result Display: Shows the predicted value and an accompanying graph.
Contributing
Feel free to contribute to this project by creating pull requests or submitting issues.
