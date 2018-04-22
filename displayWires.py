#This will then go through and plot the wires.
import numpy as np

import sys
import matplotlib.pyplot as plt

desireFile = 'text_actu.npy' 
outputFile = 'text_pred.npy'
inputFile = 'text_orig.npy'


desire = np.load(desireFile)
#print(desire[0])
#print(desire)
output = np.load(outputFile)
inputs = np.load(inputFile)
print("Input first:")
print(inputs[0,1:6,:])
print(output[0,1:6,:])
print(desire[0,1:6,:])

plt.plot(output[0][:],color='black',label='Predictions') #These are the steps for the first batch file.
plt.plot(desire[0][:],color='red',label='True Goal')
plt.plot(inputs[0][:],color='blue',label='Input')



plt.legend(loc='best')
plt.show()
