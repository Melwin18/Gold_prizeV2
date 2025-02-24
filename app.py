from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from ml_model import validate_model, load_data, train_model, predict_future_prices  # Import the ML model function
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.config['EXPLAIN_TEMPLATE_LOADING'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Home Page
@app.route('/')
def home():
    return render_template('new_index.html')

# Price Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        date = request.form['date']
        # Convert date to a format suitable for the model
        # (You may need to preprocess the date further)
        prediction = predict_price(date)
        return render_template('predict.html', prediction=prediction)
    return render_template('predict.html')

# Import Data Visualization Page
@app.route('/import-data')
def import_data():
    # Load import data from Excel files
    data_2024 = pd.read_excel('data/data_2024.xls')
    data_2023_2024 = pd.read_excel('data/data_2023-2024.xls')
    data_2025 = pd.read_excel('data/data_2025.xls')

    # Combine data for visualization
    combined_data = pd.concat([data_2024, data_2023_2024, data_2025])
    chart_data = combined_data.to_dict(orient='records')

    return render_template('import_data.html', chart_data=chart_data)

# Market Analysis Page
@app.route('/market-analysis')
def market_analysis():
    try:
        print("Loading data...")
        data = load_data(file_type='prediction')
        print(f"Data loaded with shape: {data.shape}")
        print(f"Data columns: {data.columns}")
        
        latest_price = data['USD_Price'].iloc[-1]
        week_avg = data['USD_Price'].tail(7).mean()
        print(f"Latest price: {latest_price}, Weekly average: {week_avg}")

        # Fetch news data
        api_url = "https://newsdata.io/api/1/news?apikey=pub_70847c90b783a2a9c8f9e456116416271f879&q=gold&country=in&language=en&category=business"
        response = requests.get(api_url)
        if response.status_code == 200:
            news_data = response.json()
            print(f"News data fetched successfully: {news_data}")
        else:
            news_data = {"error": "Failed to fetch news"}
            print("Failed to fetch news")

        return render_template('analysis.html',
                               latest_price=latest_price,
                               week_avg=week_avg,
                               news_data=news_data)
    except Exception as e:
        print(f"Error in market_analysis: {str(e)}")
        return render_template('error.html', message=str(e)), 500

# Price Prediction Page
@app.route('/price-prediction', methods=['GET', 'POST'])
def price_prediction():
    data = load_data('prediction')
    if request.method == 'POST':
        country = request.form.get('country')
        model, _ = train_model(data, country)
        future_prices = predict_future_prices(model, data[data['Country'] == country], future_days=30)
        return render_template('price_prediction.html', future_prices=future_prices, country=country)
    else:
        countries = data['Country'].unique()
        return render_template('price_prediction.html', countries=countries)

# Data Visualization Page
@app.route('/data-visualization')
def data_visualization():
    return render_template('data_visualization.html')

# News Section
@app.route('/news')
def news():
    # Fetch news data from the API
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        return render_template('news.html', news=[], error='API key not configured')
    
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q=gold&country=in&language=en&category=business"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json().get('results', [])
        return render_template('news.html', news=news_data)
    except requests.exceptions.RequestException as e:
        return render_template('news.html', news=[], error=str(e))

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)