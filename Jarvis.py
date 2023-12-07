import random
import time
import json
import torch
import torch.nn as nn
import re
from Brain import NeuralNet
from Speak import Say
from Task import NonInputExecution, InputExecution
from NeuralNetwork import bag_of_words, stem, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")
with open("intents.json", "r") as f:
    intents = json.load(f)
FILE = "TrainData.pth"
data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
# ____________________________________________________________________________________________________________#

Name = "Jarvis"
from Listen import Listen

checktrue = True


def Main():
    sentence = Listen()
    results = str(sentence)
    exit_patterns = [
        r".*bye.*",
        r".*stop.*",
        r".*ok leave now.*",
        r".*ok leave.*",
        r".*ok go now.*",
        r".*ok stop now.*",
        r".*ok stop.*",
        r".*ok go to sleep.*",
        r".*ok go to sleep now.*",
        r".*go to sleep.*",
        r".*you can go now.*",
        r".*bye-bye.*",
        r".*goodbye.*",
        r".*good bye.*",
        r".*bye bye.*",
        r".*you can leave now.*",
        r".*you can leave.*",
        r".*please leave.*",
        r".*you can go.*",
        r".*can you leave.*",
        r".*can you go.*",
        r".*can you stop.*",
        r".*can you go to sleep.*",
        r".*can you go now.*",
        r".*can you leave now.*",
        r".*can you go to sleep now.*",
        r".*you are free to leave.*",
        r".*free to leave.*",
        r".*free to go.*",
        r".*you are free to go.*",
        r".*you are free to go now.*",
        r".*you are free to leave now.*",
    ]

    if any(re.match(pattern, sentence) for pattern in exit_patterns):
        global checktrue
        checktrue = False
        responses = [
            response
            for intent in intents["intents"]
            if intent["tag"] == "bye"
            for response in intent["responses"]
        ]
        exit_response = random.choice(responses)
        print(f"{Name}: {exit_response}")
        Say(exit_response)
        exit()
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                if "time" in reply:
                    NonInputExecution(reply)
                elif "date" in reply:
                    NonInputExecution(reply)
                elif "joke" in reply:
                    print(reply)
                    NonInputExecution(reply)
                elif "fact" in reply:
                    print(reply)
                    NonInputExecution(reply)
                elif "day" in reply:
                    NonInputExecution(reply)
                elif "sscreenshot" in reply:
                    NonInputExecution(reply)
                elif "wikipedia" in reply:
                    InputExecution(reply, results)
                elif "google" in reply:
                    InputExecution(reply, results)
                elif "Notepad" in reply:
                    InputExecution(reply, sentence)
                elif "cmd" in reply:
                    InputExecution(reply, sentence)
                elif "powershell" in reply:
                    InputExecution(reply, sentence)
                elif "location" in reply:
                    InputExecution(reply, sentence)
                elif "weather" in reply:
                    InputExecution(reply, sentence)
                elif "screenshot" in reply:
                    InputExecution(reply, sentence)

                else:
                    Say(reply)
    elif results == "" or results == None:
        Say("Sorry, I didn't get that. Please say it again.")
    else:
        print(f"{Name}: This is what i found on web")
        Say("This is what i found on web")
        InputExecution("google", results)

while checktrue:
    Main()
