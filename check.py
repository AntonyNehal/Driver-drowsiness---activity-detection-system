import requests
import pyttsx3

API_KEY = "AIzaSyAF_32ESHmOqRTtosRZXQy4OHKDIT--ypA"

def get_current_location():
    """Fetches latitude and longitude using an IP-based API."""
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return data.get('lat'), data.get('lon')
    except:
        return None, None

def get_nearby_places(lat, lng, place_type):
    """Fetches nearby places (hotels, restrooms) using Google Places API."""
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=10000&type={place_type}&key={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        return [place["name"] for place in data.get("results", [])] or ["No places found nearby"]
    except:
        return ["Error fetching places"]

def speak_nearby_places(places, place_type):
    """Speaks the list of nearby places."""
    engine = pyttsx3.init()
    engine.say(f"Nearby {place_type}s are:")
    
    for place in places:
        engine.say(place)
    
    try:
        engine.runAndWait()
    except RuntimeError:
        print("Speech engine error: run loop already started")

# Main execution
lat, lng = get_current_location()
if lat and lng:
    print(f"Latitude: {lat}, Longitude: {lng}")
    
    hotels = get_nearby_places(lat, lng, "lodging")
    restrooms = get_nearby_places(lat, lng, "restroom")

    print("Nearby Hotels:", hotels)
    print("Nearby Restrooms:", restrooms)

    speak_nearby_places(hotels, "hotel")
    speak_nearby_places(restrooms, "restroom")
else:
    print("Failed to fetch location.")
