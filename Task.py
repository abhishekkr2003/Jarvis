import time
import datetime
from Speak import Say
import requests
from gtts import gTTS
from PIL import ImageGrab
import pyjokes
import requests
from geopy.geocoders import Nominatim
import pandas as pd
from bs4 import BeautifulSoup
def Time():
    current_time = datetime.datetime.now().strftime("%I:%M %p")  # Adds AM/PM format
    Say("The time is " + current_time)


def get_amazing_fact():
    url = "http://numbersapi.com/random/trivia?json"
    response = requests.get(url)
    
    if response.status_code == 200:
        fact = response.json()["text"]
        return fact
    else:
        return "Failed to retrieve an amazing fact."



def Date():
    current_date = datetime.datetime.now().strftime("%d %B %Y")
    Say("The date is " + current_date)
def Day():
    current_day = datetime.datetime.now().strftime("%A")
    Say("The day is " + current_day)
def get_random_joke():
    joke = pyjokes.get_joke()
    return joke
def NonInputExecution(query):
    
    query = str(query)
    query = query.lower()
    if 'time' in query:
        Time()
    elif 'date' in query:
        Date()
    elif 'day' in query:
        Day()
    elif "jokes" in query:
        random_joke = get_random_joke()
        print(random_joke)
        Say("Here is a joke for you master , "+ random_joke)
    elif "fact" in query:
        amazing_fact = get_amazing_fact()
        print(amazing_fact)
        Say("Here is an amazing fact for you master ,  "+ amazing_fact)
        
def Wikipedia(tag,query):
    name = str(query.replace("",""))
    import wikipedia
    result = wikipedia.summary(name,sentences=2)
    Say(result)

def InputExecution(tag, query):
    import wikipedia
    if "wikipedia" in tag:
        name = str(query)
        try:
            result = wikipedia.summary(name)
            Say(result)
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation errors (multiple matching results)
            Say(f"Multiple results found. Please specify your query more.")
        except wikipedia.exceptions.HTTPTimeoutError as e:
            # Handle HTTP timeout errors (network or Wikipedia server issue)
            Say(f"Sorry, I couldn't retrieve the information. Please try again later.")
        except wikipedia.exceptions.PageError as e:
            # Handle page errors (no matching results)
            Say(f"Sorry, no results found for your query.")
    if "google" in tag:
        import webbrowser
        name = str(query)
        url = f"https://www.google.com/search?q={name}"
        webbrowser.open(url)
    if "Notepad" in tag:
        import os
        path = "C:\\Program Files\\WindowsApps\\Microsoft.WindowsNotepad_11.2307.27.0_x64__8wekyb3d8bbwe\\Notepad\\Notepad.exe"
        os.startfile(path)
    if "cmd" in tag:
        import os
        path = "C:\\Users\\Admin\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\System Tools\\Command Prompt.lnk"
        os.startfile(path)

    if "powershell" in tag:
        import os
        path ="C:\\Users\\Admin\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Windows PowerShell\\Windows PowerShell.lnk"
        os.startfile(path)
    if "location" in tag:
        Say("Please wait Sir, I am finding your location")
        try:
            ipAdd = requests.get('https://api.ipify.org').text
            print(ipAdd)
            url = 'https://get.geojs.io/v1/ip/geo/'+ipAdd+'.json'
            geo_requests = requests.get(url)
            geo_data = geo_requests.json()
            city = geo_data['city']
            state = geo_data['region']
            country = geo_data['country']
            Say(f"Master, I believe we find ourselves in the city of {city}, in the state of {state}, which is a part of  country, {country}. I would suggest double-checking our location, just to be sure.")
        except Exception as e:
            Say("Sorry Sir, Due to network issue I am not able to find your location")
            pass
 
    if "weather" in tag:
        def get_location():
    # Get current location based on IP address
            ip_address = requests.get('https://api.ipify.org').text
            print(ip_address)
            url = 'https://get.geojs.io/v1/ip/geo/' + ip_address + '.json'
            geo_requests = requests.get(url)
            geo_data = geo_requests.json()
            city = geo_data['city']
            return city

        def get_weather_info(city):
            try:
                url = "https://www.google.com/search?q=" + "weather " + city
                html = requests.get(url).content
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
                str_data = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text
                data = str_data.split('\n')
                sky_condition = data[1]
                return f"The current temperature in {city} is {temp} and sky is {sky_condition}"

            except Exception as e:
                print(f"An error occurred: {e}")
                return "Sorry Sir, Due to network issues, I am not able to find your location."

# Example usage
        Say("Please wait Sir, I am finding the weather conditions")
        city_name = get_location()
        weather_info = get_weather_info(city_name)
        Say(weather_info)
        
    if "screenshot" in tag:
        import os
        Say("Taking screenshot")
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        screenshot = ImageGrab.grab()
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screenshot_{timestamp}.png"
        folder_path = "ScreenShots"
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        screenshot.save(os.path.join(folder_path, filename))
        Say("Screenshot saved as " + filename)    
    