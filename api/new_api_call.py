import requests

url = 'http://localhost:8000/predict'

# these are turned into strings somehow maybe? should be fed to the model differently?
params = {
    'user_id': 1337,
    'top_n': 10
}

response = requests.get(url, params=params)

# debugging
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: Received status code {response.status_code}")
    print(response.text)
