# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
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
			streetNames.add(row['STREET'])
	global aList
	aList=list(streetNames)
	#print aList

def readStreetNames():
	with open('StreetNames_v1.csv') as csvfile:
		reader=csv.DictReader(csvfile)
		streetNames=set()
		for row in reader:
			#print(row['STREET'])
			streetNames.add(row['STREET'])
	global aList
	aList=list(streetNames)
	return aList

def writeNames(aList):
	with open('StreetNames1.csv', 'w') as f:
    		for s in aList:
        		f.write(s.encode("UTF-8") + '\n')


def detectEntities(testSentence):
	#testSentence = detectStreetAdd(testSentence)
	
	testSentence = RTag.tag(testSentence)
	print(testSentence)
	if "<SIZE>" not in testSentence:
		return ""

	regex = "(<SIZE>)"
	reg = re.compile(regex, re.IGNORECASE)
	matches = re.finditer(reg, testSentence)

	entList=[]
	for match in matches:
		endIndex = match.end()
		s=testSentence[endIndex:]
		s=s.split("</SIZE>")[0]
		entList.append(s)
		print("Found Area: ",s)

	#s=testSentence.split("<SIZE>")[1]
	#s=s.split("</SIZE>")[0]
	
	#print(s)
	return entList
	#print(testSentence)

def detectStreetAdd(testSentence):
	foundItems=[]
	for item in aList:
		if item in testSentence:
			foundItems.append(item)
	for addr in foundItems:
		testSentence = re.sub(addr + '(?!</ADDR>)', '<ADDR>' + addr + '</ADDR>', testSentence)
	return testSentence	

def detectDate(testSentence):
	testSentence = RTag.tag(testSentence)
	#print(taggedSentence)
	if "<DATE>" not in testSentence:
		return ""
	regex = "(<DATE>)"
	reg = re.compile(regex, re.IGNORECASE)
	matches = re.finditer(reg, testSentence)
	entList=[]
	for match in matches:
		endIndex = match.end()
		s=testSentence[endIndex:]
		s=s.split("</DATE>")[0]
		entList.append(s)
		print("Found Area: ",s)
	
	#print(s)
	return entList
	#return None

def detectArea(testSentance):
	return None
	

readStreetNames()
#detectEntities('I move from  Blockdammweg to Am Wiesenrain which is 500 m2 in area. I will move on 22 September.')
#s = unicode('I intend to buy a new house at Dreysestraße, Berlin 500 square meters, what will be my new house insurance prices.',"utf-8")
#s = unicode('I have my new tenant from next week in my current house. Some details : Area 70 m2 and Location: Taufsteinweg 21, 11023 Berlin, Germany',"utf-8")
#s = 'I am in urgent need to be insured about my new big 300 meters squares house on BorodinStrasse starting 1st January'

#print s
#s=detectEntities(s)
#detectEntities(sys.argv[1])
#writeNames(aList)
