from flask import Flask, render_template, jsonify
import pyaudio
import wave
import time
import os
import threading

app = Flask(__name__, template_folder="templates")

recording = False

#Recording function, records samples of type label
def record(label):
    audio = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3

    sample = 0
    global recording
    recording = True

    dirPath = f"webserverDataset/{label}"
    os.makedirs(dirPath, exist_ok=True) 

    while recording:

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()

        fileName = f"sample_{int(time.time() * 1000)}_{sample+1:03d}.wav"
        waveFile = wave.open(os.path.join(dirPath, fileName), 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
        sample += 1    

    audio.terminate()

@app.route("/")
def home():
    return render_template("index.html")

#For recording audio samples
@app.route("/label/<label>", methods = ['GET'])
def readLabel(label):
    #Start a new thread for the recording
    if not recording:
        thread = threading.Thread(target=record, args=(label,))
        thread.start()
    return jsonify({"status": f"Recording {label}"})

#For stopping the audio recording
@app.route("/stop", methods=['GET'])
def stop_recording():
    global recording
    recording = False 
    return jsonify({"status": "Recording stopped"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
