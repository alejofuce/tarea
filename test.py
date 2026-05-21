import requests

url = "http://localhost:5000/predict"
payload = {"features": [5.1, 3.5, 1.4, 0.2]}

response = requests.post(url, json=payload)
print(response.json())
