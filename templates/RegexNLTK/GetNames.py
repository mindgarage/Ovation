#Reconizing Street Names in Berlin, Dates and Size tags for House Insurance Use Case
# -*- coding: utf-8 -*-

import csv
import timex as RTag
import re
import sys
def readStreetAdd():
	with open('berlin.csv') as csvfile:
		reader=csv.DictReader(csvfile)
		streetNames=set()
		for row in reader:
			#print(row['STREET'])
			streetNames.add(unicode(row['STREET'],"utf-8"))
	global aList
	aList=list(streetNames)
	#print aList

def readStreetNames():
	with open('StreetNames_v1.csv') as csvfile:
		reader=csv.DictReader(csvfile)
		streetNames=set()
		for row in reader:
			#print(row['STREET'])
			streetNames.add(unicode(row['STREET'],"utf-8"))
	global aList
	aList=list(streetNames)

def writeNames(aList):
	with open('StreetNames1.csv', 'w') as f:
    		for s in aList:
        		f.write(s.encode("UTF-8") + '\n')


def detectEntities(testSentence):
	testSentence = detectStreetAdd(testSentence)
	testSentence = RTag.tag(testSentence)
	#area = RTag.tag(testSentence)
	print testSentence

def detectStreetAdd(testSentence):
	foundItems=[]
	for item in aList:
		if item in testSentence:
			foundItems.append(item)
	for addr in foundItems:
		testSentence = re.sub(addr + '(?!</ADDR>)', '<ADDR>' + addr + '</ADDR>', testSentence)
	return testSentence	

def detectDate(testSentence):
	return None

def detectArea(testSentance):
	return None
	

readStreetNames()
#detectEntities('I move from  Blockdammweg to Am Wiesenrain which is 500 m2 in area. I will move on 22 September.')
s = unicode('I need an insurance for my new big house of 500 square meters where I will move from 15/12/17 from at Lemgoer Strasse to Luchweg',"utf-8")
#print s
detectEntities(s)
#detectEntities(sys.argv[1])
#writeNames(aList)
