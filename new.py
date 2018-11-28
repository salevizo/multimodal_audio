from pyAudioAnalysis.audioSegmentation import silenceRemoval as sR 
from pyAudioAnalysis.audioBasicIO import readAudioFile
import os,readchar
import tkSnack
input_file = "/home/cer/Desktop/multimodal/multimodal_audio/sounds/KobeBryant.wav"
fs, x = readAudioFile(input_file)
seg_lims = sR(x, fs, 0.05, 0.05, 0.05, 0.5, True)
# play each segment:
for i_s, s in enumerate(seg_lims):
    print("Playing segment {0:d} of {1:d} "
          "({2:.2f} - {3:.2f} secs)".format(i_s, len(seg_lims), s[0], s[1]))
    # save the current segment to temp.wav
    os.system("ffmpeg -i {} -ss {} -t {} temp.wav "
              "-loglevel panic -y".format(input_file, s[0], s[1]-s[0]))
    # play segment and wait for input
    s = tkSnack.Sound(load='temp.wav') 
    s.play()
    #os.system("play temp.wav")
    readchar.readchar()
