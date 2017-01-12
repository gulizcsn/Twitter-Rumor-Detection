from __future__ import print_function
import json
import re
from nltk.corpus import stopwords

x=0
global y 
y = long(x)

def extractTweetsToFile(input,output):
	f = open(output,'w')
	with open(input, 'r') as inFile:
		
		for line in inFile:
			jsonData = json.loads(line)

			if(jsonData["_source"]["lang"]!="en"):
				continue
			text = jsonData["_source"]["text"]
			raw = text.replace('\n','');
			f.write(raw.encode('utf-8')+"\n")

	f.close()


def printOutJsonLines(fileName,count):
	y =0
	with open(fileName, 'r') as inFile:
		
		for line in inFile:

			y+=1
			if(y==count):
				return
			else:
				print(line)


def printFirstLines(input,limit):
	counter = 0
	with open(input, 'r') as inFile:
		
		for line in inFile:
			
			print(line,end="")
			
			if(counter==limit):
				return
			counter+=1

	f.close()

#input is the file received from jsonFile
def removeNonLatin(input,output):

	y = 0
	emoji_pattern = re.compile(r'[^\u0000-\u007f\s]+', flags=re.UNICODE)
	pattern_list = [ur'[^\x00-\x7F]',r'#\b\w+\b',r'@\b\w+\b',r'[rt]+(\s+\.*@\w+:)+',\
	r'(https:?\/*[a-z0-9A-Z.\/]*)',r'http:+[\w+\/+.+-]+\b',r'http\b',r'\bhtt\b',r'\bht\b',r'\brt\b',r'&\w+.+?',r'[^a-z0-9A-Z_ ]',r'\b\w{1,2}\b']
	#result = re.sub(regex_pat, u'', text) 
	f = open(output,'w')
	with open(input, 'r') as inFile:
		
		for line in inFile:

			result = line.lower()
			for regex_pat in pattern_list:
				result = re.sub(regex_pat, u' ', result)
			f.write(result.strip()+"\n")


	f.close()

def removeDublicates(input,output):

	f = open(output,'w')

	with open(input) as inFile:
	    seen = set()
	    for line in inFile:
	        if not line in seen:
	        	f.write(line)
	        	seen.add(line)


	f.close()




# run this program to extract tweets from the json files into removed_dublicates_en.txt
extractTweetsToFile('all-tweets.jsonl','raw_lines_en.txt')
removeNonLatin('raw_lines_en.txt','clean_data_en.txt')
removeDublicates('clean_data_en.txt','removed_dublicates_en.txt')




#printOutLines('all-tweets.jsonl','raw_json.txt')

#printFirstLines('raw_json.txt',400)
#print y
#removeDublicates('clean_tweets.txt','dublicates_removed.txt')
#removeLinks('raw_json.txt','raw_latin.txt')
#printOutJsonLines('all-tweets.jsonl',100)


#printFirstLines('removed_dublicates_en.txt',10)
#printFirstLines('raw_lines_en.txt',10)










