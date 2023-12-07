import speech_recognition as sr

def Listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1.5  # Increase the pause threshold to 1.5 seconds
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print("You said: " + query)
        query = str(query)
        return query.lower()
    except Exception as e:
        print(e)
        print("Could not recognize audio say that again...")
        return "None"
