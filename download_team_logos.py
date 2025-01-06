import os
import requests
from PIL import Image
from io import BytesIO

# Create logos directory if it doesn't exist
if not os.path.exists('logos'):
    os.makedirs('logos')

# Dictionary of team abbreviations and their logo URLs
team_logos = {
    'LAR': 'https://static.www.nfl.com/image/private/f_auto/league/ayvwcmluj2ohkdlbiegi',
    'KC': 'https://static.www.nfl.com/image/private/f_auto/league/ujshjqvmnxce8m4obmvs',
    'TEN': 'https://static.www.nfl.com/image/private/f_auto/league/pln44vuzugjgipyidsre',
    'IND': 'https://static.www.nfl.com/image/private/f_auto/league/ketwqeuschqzjsllbid5',
    'WAS': 'https://static.www.nfl.com/image/private/f_auto/league/xymxwrxtyj9fhaemhdyd',
    'ATL': 'https://static.www.nfl.com/image/private/f_auto/league/d8m7hzpsbrl6pnqht8op',
    'PHI': 'https://static.www.nfl.com/image/private/f_auto/league/puhrqgj71gobgdkdo6uq',
    'PIT': 'https://static.www.nfl.com/image/private/f_auto/league/xujg9t3t4u5nmjgr54wx',
    'CAR': 'https://static.www.nfl.com/image/private/f_auto/league/ervfzgrqdpnc7lh5gqwq',
    'NO': 'https://static.www.nfl.com/image/private/f_auto/league/grhjkahghjkk17v43hdx',
    'MIN': 'https://static.www.nfl.com/image/private/f_auto/league/teguylrnqqmfcwxvcmmz',
    'BUF': 'https://static.www.nfl.com/image/private/f_auto/league/giphcy6ie9mxbnldntsf',
    'BAL': 'https://static.www.nfl.com/image/private/f_auto/league/ucsdijmddsqcj1i9tddd',
    'HOU': 'https://static.www.nfl.com/image/private/f_auto/league/bpx88i8nw4nnabuq0oob',
    'NYG': 'https://static.www.nfl.com/image/private/f_auto/league/t6mhdmgizi6qhndh8b9p',
    'CIN': 'https://static.www.nfl.com/image/private/f_auto/league/okxpteoliyayufypqalq',
    'DET': 'https://static.www.nfl.com/image/private/f_auto/league/ocvxwnapdvwevupe4tpr',
    'LAC': 'https://static.www.nfl.com/image/private/f_auto/league/ayvwcmluj2ohkdlbiegi',
    'CLE': 'https://static.www.nfl.com/t_headshot_desktop_2x/f_auto/league/api/clubs/logos/CLE',
    'CHI': 'https://static.www.nfl.com/image/private/f_auto/league/ra0poq2ivwyahbaq86d2',
    'JAX': 'https://static.www.nfl.com/image/private/f_auto/league/qycbib6ivrm9dqaexryk',
    'LV': 'https://static.www.nfl.com/image/private/f_auto/league/gzcojbzcyjgubgyb6xf2',
    'SF': 'https://static.www.nfl.com/t_headshot_desktop_2x/f_auto/league/api/clubs/logos/SF'
}

def download_and_resize_logo(team, url, size=(50, 50)):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGBA')
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save the resized image
        img.save(f'logos/{team}.png')
        print(f"Downloaded and resized logo for {team}")
    except Exception as e:
        print(f"Error downloading logo for {team}: {str(e)}")

# Download and resize all logos
for team, url in team_logos.items():
    download_and_resize_logo(team, url)

print("Logo download and resizing complete") 