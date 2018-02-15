#This will then go through and plot the wires.
import numpy as np

import sys
import matplotlib.pyplot as plt

desireFile = 'text_actu.npy' 
outputFile = 'text_pred.npy'
inputFile = 'text_orig.npy'


desire = np.load(desireFile)
print(desire)
output = np.load(outputFile)
inputs = np.load(inputFile)

plt.plot(output[0],color='black',label='Predictions') #These are the steps for the first batch file.
plt.plot(desire[0],color='red',label='True Goal')
plt.plot(inputs[0],color='blue',label='Input')



plt.legend(loc='best')
plt.show()
