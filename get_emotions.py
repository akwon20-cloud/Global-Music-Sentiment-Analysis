import pandas as pd
import numpy as np
import json
import os
from googletrans import Translator
from transformers import BertTokenizer
from model import BertForMultiLabelClassification


all_countries = ['United States', 'United Kingdom', 'Argentina', 'Australia', 'Austria', 'Belarus', 'Belgium', 'Bolivia', 'Brazil', 'Bulgaria', 'Canada', 
                 'Chile', 'Colombia', 'Costa Rica', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Finland', 
                 'France', 'Germany', 'Greece', 'Guatemala', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Kazakhstan', 
                 'Latvia', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Morocco', 'Netherlands', 'New Zealand', 'Nicaragua', 'Nigeria', 'Norway', 'Pakistan', 'Panama', 
                 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia', 'Saudi Arabia', 'Singapore', 'Slovakia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 
                 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Venezuela', 'Vietnam']


# helper function to translate text into English
def translate_text(input_text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(input_text, dest=target_language)
    return translated_text.text

# labels
id2label = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrasment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
label2id = {}
for i, label in enumerate(id2label):
    label2id[label] = i

# Tokenizer and BERT models
# Source: https://github.com/monologg/GoEmotions-pytorch/tree/master
tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

# create a cache to save work
if os.path.exists("data/emotions/cache.json"):
    with open("data/emotions/cache.json", "r") as f:
        cache = json.load(f)
else:
    cache = {}


for country in all_countries:
    print(f"on country: {country}")
    
    df = pd.read_csv(f"data/lyrics/lyrics_{country}.csv")
    # df['lyrics'] = df['lyrics'].map(translate_text) ## no translations bc im translation api bottlenecked oop

    out = []

    for i, row in df.iterrows():
        print(i)
        artist_title = row["artist_title"]

        if artist_title not in cache:
            sentences = row['lyrics'].split('\n')
            
            # remove all translation credit lines at the top of each lyrics
            numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

            first_line = sentences[0]
            for num in numbers:
                if num in first_line:
                    sentences.pop(0)
                    break

            # remove empty lines from songs (potential idea: we can split on these instead as these are paragraph separators)
            temp = []
            for sentence in sentences:
                if sentence != "":
                    temp.append(sentence)

            sentences = temp

            # run the model on the sentences
            song_emotions = []
            threshold = 0.3
            for sentence in sentences:
                try:
                    tokenized_input = tokenizer(sentence, return_tensors='pt')
                    outputs = model(**tokenized_input)[0].detach().numpy()


                    scores = 1 / (1 + np.exp(-outputs))  # Sigmoid
                    results = []

                    for item in scores:
                        labels = []
                        scores = []
                        for idx, s in enumerate(item):
                            if s > threshold:
                                labels.append(id2label[idx])
                                scores.append(s)
                        results.append({"labels": labels, "scores": scores})
                    
                    song_emotions.append(results[0]["labels"])
                except:
                    song_emotions.append([])

            # add the 'out'
            out.append(song_emotions)

            # add to cache
            cache[artist_title] = song_emotions
        else:
            song_emotions = cache[artist_title]
            out.append(song_emotions)


    with open("data/emotions/cache.json", "w") as f:
        json.dump(cache, f)

    out_series = pd.Series(out)
    out_series.name = "emotions"

    out_df = pd.concat([df, out_series], axis=1)
    out_df.to_csv(f"data/emotions/emotions_{country}.csv")