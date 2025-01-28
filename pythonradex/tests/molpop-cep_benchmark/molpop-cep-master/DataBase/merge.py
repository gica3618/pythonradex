#/usr/bin/env python

import glob

files = glob.glob('*.lev')

for f in files:
	str = f.split('.')
	
	file = open(str[0]+".lev","r")
	linesLev = file.readlines()
	file.close()
	
	file = open(str[0]+".aij","r")
	linesAij = file.readlines()
	file.close()
	
	file = open(str[0]+".molecule","w")
	linesLev.append("\n")
	linesLev.append("\n")
	
	for i in linesAij:
		linesLev.append(i)
	
	for i in linesLev:
		file.write(i)
		
	file.close()