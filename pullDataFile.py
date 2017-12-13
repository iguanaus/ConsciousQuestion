from numpy import genfromtxt
import numpy 

#Now only edit is going to be selection of wires.


def getSnipData(filename,label,data_len=10000):
	#Pull and process the snip 2 A
	goodWires = [4,5,6,8,10,12]
	my_data = genfromtxt(filename,delimiter='\t')
	print("The number of lines are: ", len(my_data))
	print("Pulling between lines: " , int(len(my_data)*0.25))
	print("Up to the max of line: " , int(len(my_data)*0.75))
	beginInd = int(len(my_data)*0.25)
	endInd = min(beginInd + data_len, int(len(my_data)*0.75))
	my_answer = numpy.full((endInd-beginInd, 1), label)
	print("Final pull from : " , beginInd, "to", endInd)
	my_data = my_data[beginInd:endInd]
	my_data_x = my_data[:,goodWires]
	return (my_data,my_answer)

data_1, ans_1 = getSnipData("/mnt/data0/euler/m00038/Raw/snippet-2_A.csv",1)
data_2, ans_2 = getSnipData("/mnt/data0/euler/m00038/Raw/snippet-12_D.csv",2)
data_3, ans_3 = getSnipData("/mnt/data0/euler/m00038/Raw/snippet-22_A.csv",1)
data_4, ans_4 = getSnipData("/mnt/data0/euler/m00038/Raw/snippet-21_D.csv",1)

finalData = numpy.concatenate((data_4,data_3,data_2, data_1), axis=0)
finalAns = numpy.concatenate((ans_4,ans_3,ans_2,ans_1),axis=0)

print(finalAns)
print(finalData)

numpy.savetxt('snip_data.csv', finalData, delimiter='\t',fmt='%.2f')
numpy.savetxt('snip_lables.csv',finalAns,delimiter='\t',fmt='%d')




