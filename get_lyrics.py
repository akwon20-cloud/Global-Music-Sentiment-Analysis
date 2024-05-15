import pandas as pd
from dotenv import load_dotenv
import os
import requests
from requests.exceptions import Timeout
import json
import lyricsgenius
import pickle

load_dotenv()

genius_client_id = os.getenv("genius_client_id")
genius_secret = os.getenv("genius_secret")
genius_token = os.getenv("genius_token")


all_countries = ['United States', 'United Kingdom', 'Argentina', 'Australia', 'Austria', 'Belarus', 'Belgium', 'Bolivia', 'Brazil', 'Bulgaria', 'Canada', 
                 'Chile', 'Colombia', 'Costa Rica', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Finland', 
                 'France', 'Germany', 'Greece', 'Guatemala', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Kazakhstan', 
                 'Latvia', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Morocco', 'Netherlands', 'New Zealand', 'Nicaragua', 'Nigeria', 'Norway', 'Pakistan', 'Panama', 
                 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia', 'Saudi Arabia', 'Singapore', 'Slovakia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 
                 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Venezuela', 'Vietnam']


url = "https://api.genius.com/search"
headers = {"Authorization": f"Bearer {genius_token}"}
genius = lyricsgenius.Genius(genius_token)

cache = {} # check if cache.json exists, if not set cache = {}

for country in all_countries:
    print(f"on country: {country}")

    filename = f"data/topsongs/6.S079 Songs - {country}.csv"
    df = pd.read_csv(filename)

    out = []

    for i, row in df.iterrows():
        print(i)
        artist_title = row["artist_title"]

        if artist_title not in cache:
            # get the song url from genius
            while True:
                try:    
                    params = {"q": artist_title}
                    response = requests.get(url=url, params=params, headers=headers)
                    response_dict = json.loads(response.text)
                    song_url = response_dict["response"]["hits"][0]["result"]["url"]

                    # get lyrics
                    lyrics = genius.lyrics(song_url=song_url, remove_section_headers=True)

                    # add to 'out'
                    d = {"artist_title": artist_title, "lyrics": lyrics}
                    out.append(d)

                    # cache result
                    cache[artist_title] = lyrics

                    break
                except Timeout:
                    print("timed out, retrying")
                    continue  
                except Exception as e:
                    # generally, this means that genius couldn't find our song, so we skip this song
                    print(e)
                    break

        else:
            lyrics = cache[artist_title]

            # add to 'out'
            d = {"artist_title": artist_title, "lyrics": lyrics}
            out.append(d)

    with open("data/lyrics/cache.json", "w") as f:
        json.dump(cache, f)

    out_df = pd.DataFrame(out)
    out_df.to_csv(f"data/lyrics/lyrics_{country}.csv")
