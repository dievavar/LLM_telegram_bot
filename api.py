import requests
from pprint import pprint


url = "https://api.intelligence.io.solutions/api/v1/models"

headers = {
    "accept":"application/json",
    "Authorization": "AUTH_TOKEN"
}

response = requests.get(url, headers = headers)
pprint(response.json())

# for i in range(len(response.json()['data'])):
#     print(response.json()['data'][i]['id'])