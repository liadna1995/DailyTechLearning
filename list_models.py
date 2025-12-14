
import google.genai as genai

API_KEY = "AIzaSyADjrrZQrB9PXkERYJ-AxNNCMULtzxKkoE"

try:
    client = genai.Client(api_key=API_KEY)
    print("Listing available models...")
    for m in client.models.list():
        print(f"Name: {m.name}")
except Exception as e:
    print(f"Error: {e}")
