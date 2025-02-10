#/usr/bin/env python

import glob

# List all the files with collisional data and save the headers in list_collisions.dat

files = glob.glob('*.kij')

fout = open("list_collisions.dat","w")

for f in files:
	
	file = open(f,"r")
	linesColl = file.readlines()
	file.close()
	
	t = [f+'\n']
			
	for l in linesColl:
		if (l[0] != '>'):
			t.append(l)
		else:
			t.append('***********\n')			
			break
	
	for i in t:
		fout.write(i)
		
	file.close()
	
fout.close()