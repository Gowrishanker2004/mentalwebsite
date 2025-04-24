def fuse_emotions(face, voice):
    if face == voice:
        return face
    elif face in ["Sad", "Angry", "Fear"] and voice in ["Sad", "Angry", "Fear"]:
        return "Depressed"
    elif "Happy" in [face, voice]:
        return "Happy"
    elif "Sad" in [face, voice]:
        return "Sad"
    elif "Angry" in [face, voice]:
        return "Angry"
    elif "Fear" in [face, voice]:
        return "Fear"
    else:
        return "Neutral"
