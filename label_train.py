import os

a1 = open("label_trainX.txt", "w")
a2 = open("label_trainY.txt", "w")
a3 = open("label_trainZ.txt", "w")

for path, subdirs, files in os.walk('data/train'):
	for filename in files:
		f = str(filename)
		#a.write(f[:-9] + os.linesep)
		
		if f[:4] == "n015":
			a1.write(f + ";1,0,0" + os.linesep)
		if f[:4] == "n020":
			a2.write(f + ";0,1,0" + os.linesep)
		if f[:4] == "n017":
			a3.write(f + ";0,0,1" + os.linesep)
		