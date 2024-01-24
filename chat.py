import random
import json
import requests
import re
import openai
from googlesearch import search  
import webbrowser
import datetime
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from config import openai_api_key, weather_api_key, google_search_api_key # Import keys from config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

google_api_key = google_search_api_key
openai.api_key = openai_api_key

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Ava"

def extract_city_from_message(message):
    city_match = re.search(r'in (\w+)', message)
    if city_match:
        return city_match.group(1)
    else:
        return None

def get_date():
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    return current_date

def get_time():
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return current_time

import webbrowser

def perform_google_search(query):
    try:
        search_results = list(search(query, num_results=1))

        if search_results:
            first_result_url = search_results[0]
            webbrowser.open(first_result_url, new=2)  # Open URL in a new tab (or browser window)
            return "Redirecting to the first search result..."
        else:
            return "No results found on Google."
    except Exception as e:
        print(f"Error during Google search: {e}")
        return "Failed to perform Google search at the moment. Please try again later."



def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "google":
                    return perform_google_search(msg)
                elif tag in ["rephrase", "essay", "summary", "translation", "code_generation", "poetry", "storytelling", "math_problem"]:
                    response = openai_completion(msg)
                    return response
                elif tag == "weather":
                    city_name = extract_city_from_message(msg)
                    complete_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={weather_api_key}&units=metric"
                    response = requests.get(complete_url)
                    weather_data = response.json()

                    if response.status_code == 200:
                        main_weather = weather_data['weather'][0]['main']
                        description = weather_data['weather'][0]['description']
                        temp = weather_data['main']['temp']
                        wind_speed = weather_data['wind']['speed']
                        clouds = weather_data['clouds']['all']
                        pressure = weather_data['main']['pressure']
                        return f"The weather in {city_name} {temp}°C temperature, wind {wind_speed} m/s, clouds {clouds}%, {pressure} hpa"
                elif tag == "date":
                    return f"The current date is {get_date()}."
                elif tag == "time":
                    return f"The current time is {get_time()}."
                else:
                    return random.choice(intent['responses'])
    return "I do not understand..."



def openai_completion(msg, max_tokens=1000):
    prompt = f"User: {msg}\nAI:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg},
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message['content']

