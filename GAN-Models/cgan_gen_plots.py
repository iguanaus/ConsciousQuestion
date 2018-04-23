import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle

model = "MNIST"
plotDim1 = 28
plotDim2 = 28
if model == "MNIST":
	plotDim1 = 28
	plotDim2 = 28
elif model == "brain":
	plotDim1 = 1500
	plotDim2 = 10

#This plots a file saved in the out folder.

def convertFile(fileName,outName):
	samples = pickle.load(open(fileName,"rb"))

	def plot(samples):
	    fig = plt.figure(figsize=(4, 4))
	    gs = gridspec.GridSpec(4, 4)
	    gs.update(wspace=0.05, hspace=0.05)

	    for i, sample in enumerate(samples):
	        ax = plt.subplot(gs[i])
	        plt.axis('off')
	        ax.set_xticklabels([])
	        ax.set_yticklabels([])
	        ax.set_aspect('equal')
	        plt.imshow(sample.reshape(plotDim1, plotDim2),aspect='auto', cmap='Greys_r')
	        #plt.imshow(newVal,aspect='auto',cmap='Greys_r')


	    return fig


	fig = plot(samples)
	plt.savefig(outName, bbox_inches='tight')
	plt.close(fig)

#Sample
# This file goes through a directory, and each with the .dat extension and converts it. 
#
#
def convertAllFilesInDir(dir="out/"):
	for file in os.listdir(dir):
	    if file.endswith(".dat"):
	    	filePath = os.path.join(dir, file)
	    	print("Filepath: " , filePath)
	    	outName = filePath.replace('.dat','.png')
	    	convertFile(filePath,outName)

fileName = "out/000.dat"
outName = "out/000.png"

convertAllFilesInDir()
#convertFile(fileName,outName)



