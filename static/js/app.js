function fetchPrediction() {
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction-result').innerHTML = 
            `Predicted Prices: ${data.prediction.join(', ')}`;
    });
}
