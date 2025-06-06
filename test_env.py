import os
import requests

# Test if we can access the API key
api_key = os.environ.get('OPENROUTER_API_KEY')
print(f"API Key is set: {'Yes' if api_key else 'No'}")

# Try a simple API call to verify the key works
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://yourdomain.com",
    "X-Title": "TestApp"
}

try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [{"role": "user", "content": "Hello, are you working?"}]
        }
    )
    print(f"API Response Status: {response.status_code}")
    if response.status_code == 200:
        print("API call successful!")
        print(response.json())
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error occurred: {str(e)}") 