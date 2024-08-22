import speech_recognition as sr
import pyttsx3

def initialize_tts():
    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    return engine

def speak(engine, message):
    # Use text-to-speech to say the message
    engine.say(message)
    engine.runAndWait()

def listen(recognizer):
    # Listen to microphone input and convert to text
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
            return None

def main():
    recognizer = sr.Recognizer()
    tts_engine = initialize_tts()
    
    while True:
        text = listen(recognizer)
        if text:
            # Simple responses based on recognized text
            if "hello" in text.lower():
                response = "Hello! How can I help you today?"
            elif "your name" in text.lower():
                response = "I am your talking robot."
            elif "stop" in text.lower():
                response = "Goodbye!"
                speak(tts_engine, response)
                break
            else:
                response = "Sorry, I don't understand that."
            
            speak(tts_engine, response)

if __name__ == "__main__":
    main()
