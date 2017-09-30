from __future__ import absolute_import, division, print_function
# encoding=utf8
import re
import sys
#reload(sys)  
#sys.setdefaultencoding('utf8')

def findAddressTokens(address):
	regex1 = "(\d\d\d\d\d)"
	reg1 = re.compile(regex1, re.IGNORECASE)
	regex2 =  "(\d+)"
	reg2 = re.compile(regex2, re.IGNORECASE)
	regex3 = "(Berlin)"
	reg3 = re.compile(regex3, re.IGNORECASE)

	address = re.sub(r'[,|?|$|.|!]',r'',address)
	#print address

	matches = re.finditer(reg1, address)
	zipStartIndex=0
	zipStopIndex=0
	for match in matches:
		zipStartIndex=match.start()
		zipStopIndex=match.end()
	zipCode = address[zipStartIndex:zipStopIndex]
	address = address[:zipStartIndex] + address[zipStopIndex:]
	#print address

	matches = re.finditer(reg2, address)
	streetNoStartIndex=0
	streetNoStopIndex=0
	for match in matches:
		streetNoStartIndex=match.start()
		streetNoStopIndex=match.end()
	StreetNum = address[streetNoStartIndex:streetNoStopIndex]
	address = address[:streetNoStartIndex] + address[streetNoStopIndex:]
	#print address

	

	matches = re.finditer(reg3, address)
	cityStartIndex=0
	cityStopIndex=0
	for match in matches:
		cityStartIndex=match.start()
		cityStopIndex=match.end()
	address = address[:cityStartIndex] + address[cityStopIndex:]
	#print address

	StreetName = address.rstrip()
	CityName = "Berlin"

	return zipCode,StreetNum,StreetName

#zipCode,StreetNum,StreetName=findAddressTokens("Kurtschunacher Straße 16, 67663, Berlin")
#print ("zipCode: "+zipCode+" StreetNum: "+StreetNum+" StreetName: "+StreetName+" City Name:"+" Berlin")


