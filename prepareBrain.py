from numpy import genfromtxt
import numpy
import pandas as pd
#Now only edit is going to be selection of wires.
from numpy import genfromtxt
import os

file_name="NewData/snippet-25_D.csv"

my_data = genfromtxt(file_name,delimiter='\t')

print(my_data)