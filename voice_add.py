import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', 'voices[0].id')


warnings.filterwarnings("ignore")


def speak(text):
    engine.say(text)
    engine.runAndWait()


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

        try:
            statement = r.recognize_google(audio, language='en-in')
            print(f"user said:{statement}\n")

        except Exception as e:
            speak("Pardon me, please say that again")
            return "None"
        return statement

def calculate_delta(array):
   
    rows,cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(audio,rate):
       
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined



def record_audio_train():
	speak("please tell your name once again")
	Name =takeCommand()
	#Name =(input("Please Enter Your Name:"))
	for count in range(5):
		FORMAT = pyaudio.paInt16
		CHANNELS = 1
		RATE = 44100
		CHUNK = 512
		RECORD_SECONDS = 10
		device_index = 2
		audio = pyaudio.PyAudio()
		print("----------------------record device list---------------------")
		info = audio.get_host_api_info_by_index(0)
		numdevices = info.get('deviceCount')
		for i in range(0, numdevices):
			if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
				print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
		print("-------------------------------------------------------------")
		index = int(input())		
		print("recording via index "+str(index))
		stream = audio.open(format=FORMAT, channels=CHANNELS,
		                rate=RATE, input=True,input_device_index = index,
		                frames_per_buffer=CHUNK)
		print ("recording started")
		Recordframes = []
		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			Recordframes.append(data)
		print ("recording stopped")
		stream.stop_stream()
		stream.close()
		audio.terminate()
		OUTPUT_FILENAME=Name+"-sample"+str(count)+".wav"
		WAVE_OUTPUT_FILENAME=os.path.join("training_set",OUTPUT_FILENAME)
		trainedfilelist = open("training_set_addition.txt", 'a')
		trainedfilelist.write(OUTPUT_FILENAME+"\n")
		waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		waveFile.setnchannels(CHANNELS)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(Recordframes))
		waveFile.close()


def train_model():

	source  = "training_set\\"
	dest = "trained_models\\"
	train_file = "training_set_addition.txt"        
	file_paths = open(train_file,'r')
	count = 1
	features = np.asarray(())
	for path in file_paths:
		path = path.strip()
		print(path)
		sr,audio = read(source + path)
		print(sr)
		vector   = extract_features(audio,sr)
		if features.size == 0:
			features = vector
		else:
			features = np.vstack((features, vector))
		
		if count == 5:    
			gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
			gmm.fit(features)
	        
	        # dumping the trained gaussian model
			picklefile = path.split("-")[0]+".gmm"
			pickle.dump(gmm,open(dest + picklefile,'wb'))
			print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
			features = np.asarray(())
			count = 0
		count = count + 1


record_audio_train()

train_model()

#exec(open('face_Train.py').read())