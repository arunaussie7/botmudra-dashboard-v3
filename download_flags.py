import os
import requests
from PIL import Image
from io import BytesIO

# Flag image URLs (replace these with your actual flag image URLs)
FLAG_URLS = {
    'EUR': 'https://example.com/flags/eur.png',
    'GBP': 'https://example.com/flags/gbp.png',
    'CHF': 'https://example.com/flags/chf.png',
    'NZD': 'https://example.com/flags/nzd.png',
    'AUD': 'https://example.com/flags/aud.png',
    'USD': 'https://example.com/flags/usd.png',
    'JPY': 'https://example.com/flags/jpy.png',
    'CAD': 'https://example.com/flags/cad.png'
}

def download_and_save_flags():
    """Download flag images and save them to static/flags directory"""
    if not os.path.exists('static/flags'):
        os.makedirs('static/flags')
    
    for currency, url in FLAG_URLS.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Resize to 24x24 pixels
                img = img.resize((24, 24))
                img.save(f'static/flags/{currency.lower()}.png')
                print(f"Downloaded and saved {currency} flag")
            else:
                print(f"Failed to download {currency} flag: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {currency} flag: {e}")

if __name__ == "__main__":
    download_and_save_flags() 