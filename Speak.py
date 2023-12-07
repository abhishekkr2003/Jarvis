import pyttsx3

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")  # getting details of current voice
engine.setProperty("voices", voices[0].id)
engine.setProperty("rate", 170)  # setting up new voice rate


def Say(audio):
    print("")
    print(f"A.I: {audio}")
    print("")
    engine.say(audio)
    engine.runAndWait()

