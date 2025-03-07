import requests

API_KEY = "3B8488FF-278C-3D24-992A-09B987D1CAB1"
lat, lon = 37.443108, 126.7143842

url = "https://api.vworld.kr/req/address"
params = {
    "service": "address",
    "request": "getAddress",
    "format": "json",
    "crs": "epsg:4326",
    "point": f"{lon},{lat}",
    "type": "ROAD",
    "key": API_KEY
}

response = requests.post(url, data=params)
if response.status_code == 200:
    try:
        address = response.json()['response']['result'][0]['text']
        print(address)
    except:
        print(response.json())
