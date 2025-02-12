import pyaudio
import wave
  
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3

#https://realpython.com/playing-and-recording-sound-python/1
  
audio = pyaudio.PyAudio()

#open label file in append mode and set sample number to current
labels = open("misc/audioDataset/labels.txt", 'a+')
labels.seek(0)
sample = len(labels.read())

while True:
    inKey = input()
    if inKey == 'c' or inKey == 'n':

        print("Recording starting")

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        waveFile = wave.open("misc/audioDataset/sample{0}".format(sample), 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print ("Sample {0} saved".format(sample))
        
        #Write the correct label to label file
        if inKey == 'c':
            labels.write("1")
        else:
            labels.write("0")
        
        sample += 1
    
    if inKey == 'q':
        labels.close()
        audio.terminate()
        break