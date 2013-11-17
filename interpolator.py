#! /usr/bin/python
import numpy
import pandas as pd
import csv
import os

def load_file(filename):
	convertfnc = lambda x: (float)(x[-11:-9]) + ((float)(x[-8:-6])/100.) - (float)(x[-4]) + ((float)(x[5:7])-1)*24.0
	return numpy.genfromtxt(open(filename,'rb'),skip_header=1,delimiter=",",converters={0:convertfnc},dtype=[None,float or None,None,None,float or None,None],usecols=(0,1,2,3,4,5))

def intro_row(start,col, data):

	stop=start+4
	data[start+1][col] = .75*data[start][col]+0.25*data[stop][col]
	data[start+2][col] = 2.0/7.0*data[start][col] + 3.0/7.0*data[start+1][col] + 2.0/7.0*data[stop][col]
	data[start+3][col] = 1.0/9.0*data[start][col] + 2.0/9.0*data[start+1][col] + 3.0/9.0*data[start+2][col] + 3.0/9.0*data[stop][col]

def writeToFile(outname, data):
	with open(outname, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for i in range(0,len(data)):
			for j in range(0,len(data[0])):
				if(numpy.isnan(data[i][j])):
					data[i][j] = 0.0
		for i in range (0,len(data)):
			s = data[i][5]
			if(s == 0.0):
				s = ''
			writer.writerow([data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],s])

def intro_file(data):
	start = 0
	end = 4

	#this fills up ever empty row at the 15s
	for i in range(0,len(data)/4 - 1):
		intro_row(start,1, data)
		intro_row(start,2, data)
		intro_row(start,3, data)
		intro_row(start,4, data)
		start+=end
	
	#this fills up the last rows
	i = 1

	i=1
	while(start+i < len(data)):
		data[start+i][1] = data[start][1]
		data[start+i][2] = data[start][2]
		data[start+i][3] = data[start][3]
		data[start+i][4] = data[start][4]
		i+= 1

def interpolate(filename):
	output = "output.csv"
	try:
		os.remove(output)
	except OSError:
		pass
	data = load_file(filename)
	intro_file(data)
	writeToFile(output, data)
	return output