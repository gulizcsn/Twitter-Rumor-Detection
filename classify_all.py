from __future__ import print_function
import csv
import logging,string,re
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.externals import joblib
from sklearn import feature_selection
import itertools
import matplotlib.pyplot as plt
import os 
import random

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments 
op = OptionParser()


op.add_option("--confusion_matrix",
              action="store_true",
              help="Print and plot the confusion matrix.")


op.add_option("--sgd",
              action="store_true",
              help="Run SGD Classifier")

op.add_option("--linearsvc",
              action="store_true",
              help = "Run LinearSVC Classifier")
op.add_option("--ridge",
              action="store_true", 
              help="Ridge Classifier")

op.add_option("--mnb",
              action="store_true", 
              help="MultinomialNB Classifier")

op.add_option("--bernoulli",
              action="store_true", 
              help="BernoulliNB Classifier")

op.add_option("--changed",
              action="store_true", 
              help="If you change the parameters for grid search, set this")

op.add_option("--predict",
              action="store_true",
              help="after the training accept an input for prediction ")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


categories = ['alt.atheism',
'comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x',
'misc.forsale',
'rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey',
'sci.crypt',
'sci.electronics',
'sci.med',
'sci.space',
'soc.religion.christian',
'talk.politics.guns',
'talk.politics.mideast',
'talk.politics.misc',
'talk.religion.misc']

#check the number of arguments

numberOfClassifier = sum([bool(opts.sgd),bool(opts.ridge),bool(opts.linearsvc),bool(opts.mnb),bool(opts.bernoulli)])
if(numberOfClassifier!=1) :
  print("WRONG USAGE: Please provide only one classifier")
  exit()


#set the clf algorithm according to parameters passed in commandline
# the values in the paranthesis will be iterated thanks to GridSearchCV we used below
if(opts.sgd):
  classifier = SGDClassifier(n_iter=500,penalty='l2')
  parameters = {  
  'tfidf__max_df': ( 0.1,0.2,0.3,0.4 ) ,   
  'clf__alpha': (0.00005,0.0001,0.0005,0.001,0.005),
  }
  plot_label = "SGD Classifier"
  pkl_file = "SGDClassifier"

if(opts.ridge):
  classifier = RidgeClassifier(tol=1e-2, solver="sag")
  parameters = {  
  'tfidf__max_df': ( 0.1,0.2,0.3,0.4) ,   
  'clf__alpha':(0.0001,0.001,0.01,0.1,1,2,5)
  }
  plot_label = "Ridge Classifier"
  pkl_file = "RidgeClassifier"

if(opts.linearsvc):
  classifier =LinearSVC(tol=1e-3,dual=False,loss='squared_hinge')

  
  parameters = {  
  'tfidf__max_df': ( 0.1,0.2,0.3) ,
  'clf__penalty': ("l2",'l1'),
  'clf__C':(0.1,1,2,3,4,5,10,15)
  }

  plot_label = "LinearSVC Classifier"
  pkl_file = "LinearSVC"

if(opts.mnb):
  classifier = MultinomialNB()

  parameters = {  
  'tfidf__max_df': ( 0.1,0.2,0.3),   
  'clf__alpha': (0.0005,0.001,0.005,0.01)  
  }

  plot_label = "MultinomialNB Classifier"
  pkl_file = "MultinomialNB"

if(opts.bernoulli):
  classifier = BernoulliNB()


  parameters = {  
  'tfidf__max_df': ( 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),   
  'clf__alpha': (0.0005,0.001,0.005,0.01,0.05,0.01,0.1,0.5,1)   
  }

  plot_label = "BernoulliNB Classifier"
  pkl_file = "BernoulliNB"


remove = ('headers', 'footers', 'quotes')

stemmer=PorterStemmer() 
stop = set(stopwords.words('english'))

#normalize the text
def clearString(inputStr):
  #only include ascii letters and digits and space
  inputStr = re.sub(r'[^A-Za-z0-9\s]+',' ',inputStr)

  #ignore the tokens smaller than 3 characters
  inputStr = re.sub(r'\b\w{1,2}\b',' ',inputStr)
  #remove stop words  
  removed =  [i for i in inputStr.lower().split() if i not in stop]
  #stem the words to normalize them
  res = " ".join([ stemmer.stem(kw) for kw in removed])
  return res

#remove duplicate tweets
def removeDublicates(input,output):

	f = open(output,'w')

	with open(input) as inFile:
	    seen = set()
	    for line in inFile:
	        if not line in seen:
	        	f.write(line)
	        	seen.add(line)


	f.close()

#normalize tweets to append these to 20 newsgroups dataset
def clearCSV(inFile,outFile):
	#only include ascii letters and digits and space
	#inputStr = re.sub(r'[^A-Za-z0-9\s]+',' ',inputStr)

	#ignore the tokens smaller than 3 characters
	#inputStr = re.sub(r'\b\w{1,2}\b',' ',inputStr)
	emoji_pattern = re.compile(r'[^\u0000-\u007f\s]+', flags=re.UNICODE)
	pattern_list = [ur'[^\x00-\x7F]',r'#\b\w+\b',r'@\b\w+\b',r'[rt]+(\s+\.*@\w+:)+',\
	r'(https:?\/*[a-z0-9A-Z.\/]*)',r'http:+[\w+\/+.+-]+\b',r'http\b',r'\bhtt\b',r'\bht\b',r'\brt\b',r'&\w+.+?',r'[^a-z0-9A-Z_ ]',r'\b\w{1,2}\b']
	#remove stop words  
	f = open(outFile,'w')
	with open(inFile, 'rb') as fp:
		csvReader = csv.reader(fp, delimiter='\t') 
		for line in csvReader:
			result = line[3].lower() 
			for regex_pat in pattern_list:
			  result = re.sub(regex_pat, u' ', result)

			removed =  [i for i in result.lower().split() if i not in stop]
			f.write(" ".join([ stemmer.stem(kw) for kw in removed])+"\n")

	f.close()




#for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)







print("Loading 20 newsgroups dataset for categories:")

X_train = X_test = y_train = y_test = target_names = None

#check if all the training and test data already normalized, if not extract them
if os.path.isfile('data_train.txt') and os.path.isfile('data_test.txt') and os.path.isfile('y_data_test.txt') and os.path.isfile('y_data_train.txt') and os.path.isfile('target_names.txt') :
  print("Data is found and loaded from disk")


  with open("data_train.txt", "rb") as fp:   # Unpickling
    X_train = pickle.load(fp)
    
  with open("data_test.txt", "rb") as fp:   # Unpickling
    X_test = pickle.load(fp)

  with open("y_data_test.txt", "rb") as fp:   # Unpickling
    y_test = pickle.load(fp)
    #print(y_test[:10])

  with open("y_data_train.txt", "rb") as fp:   # Unpickling
    y_train = pickle.load(fp)
    #print(y_train[:10])

  with open("target_names.txt", "rb") as fp:   # Unpickling
    target_names = pickle.load(fp)
    #print(target_names[:10])

else:
  print("DATA isn't found in the disk, getting generated. This will take a while ...")

  data_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42,
                                  remove=remove)
  data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

  #normalize the data
  X_train = [clearString(text) for text in data_train.data]
  X_test = [clearString(text) for text in data_test.data]
  y_train = data_train.target 
  y_test = data_test.target
  target_names = data_train.target_names
  
  type(X_train)
  clearCSV('tweets.csv',"tempClear.txt")
  removeDublicates("tempClear.txt","tempRemoved.txt")

  newTrainData = []
  newTrainTarget = []
  newTestData = []
  newTestTarget = []


  count = 0
  with open('tempRemoved.txt', 'rb') as fp:
	for line in fp:
		count+=1
		if(count%10<8):
			newTrainData.append(line)
			newTrainTarget.append(int(13))
		else:
			newTestData.append(line)
			newTestTarget.append(int(13))


  X_train = X_train + newTrainData
  X_test = X_test + newTestData



  y_train = np.append(y_train,newTrainTarget)
  y_test = np.append(y_test,newTestTarget)

 


  for i in range(0,10):
  	print(X_train[-i],y_train[-i])

  #combined = list(zip(X_train , y_train,X_test,y_test))
  #random.shuffle(combined)
  #X_train[:], y_train[:],X_test[:],y_test[:]= zip(*combined)






  #after training save the data to load faster in the next run

  with open("data_train.txt", "wb") as fp:   # Unpickling
    pickle.dump(X_train,fp)
    
  with open("data_test.txt", "wb") as fp:   # Unpickling
    pickle.dump(X_test,fp)

  with open("y_data_test.txt", "wb") as fp:   # Unpickling
    pickle.dump(y_test,fp)

  with open("y_data_train.txt", "wb") as fp:   # Unpickling
    pickle.dump(y_train,fp)

  with open("target_names.txt", "wb") as fp:   # Unpickling
    pickle.dump(target_names,fp)

print("Data is loaded")


t0 = time()

grid_search = None

#check if the fitted model exists on the disk
#if it doesn't exist, fit the model
#if you want to regenerate the model pass --changed parameter in the command line

if os.path.isfile(pkl_file+'.pkl') and not opts.changed:
  grid_search = joblib.load(pkl_file+'.pkl') 
else:
  print("Trained data isn't found, data is trained now. this will take a while")
  #pipeline is used for sequential transformation and processing
  pipeline = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True,use_idf=True,norm='l2',binary=False)), ('clf',classifier)])



  grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1) 
  grid_search.fit(X_train, y_train)   


pred = grid_search.predict(X_test)
score = metrics.accuracy_score(y_test, pred)

print("Accuracy Score {0}".format(score))

f1_accuracy = metrics.f1_score(y_test, pred, average='macro')

print("F1 Accuracy Score {0}".format(f1_accuracy))



test_time = time() - t0

print("Time it took to do research:")
print(test_time)


print("best TfidfVectorizer parameters:")
print(grid_search.best_estimator_.named_steps['tfidf'])

print(plot_label + " parameters:")
print(grid_search.best_estimator_.named_steps['clf'])






print("\nMost significant words per category\n")
clf = grid_search.best_estimator_.named_steps['clf']

feature_names = np.array(grid_search.best_estimator_.named_steps['tfidf'].get_feature_names())

for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]

    print("%s: %s" % (label, " ".join(feature_names[top10])))

print()


#if confusion matrix is requested print it out and plot it
if(opts.confusion_matrix):
  print("confusion matrix:")
  cm = metrics.confusion_matrix(y_test, pred)
  print(cm.shape)
  for count,elem in enumerate(cm):
    for item in elem:
      print('{:>4}'.format(item),end=" ")
    print("| Total = {0} {1}".format(sum(elem),target_names[count]))
    print()


  print("Total ")

  for i in range(0,len(cm)):
    print("{:>4}".format(sum(cm[:,i])),end=" ")

  print()
  #joblib.dump(grid_search, 'LinearSVC.pkl') 

  print(metrics.classification_report(y_test, pred,
                                              target_names=target_names))



  plt.figure()
  plot_confusion_matrix(cm, classes=categories,
                        title='Confusion matrix for '+ plot_label)
  plt.show()

joblib.dump(grid_search, pkl_file + '.pkl') 


#if user wants to predict the sentences he enters, pass --predict on the command line
if(opts.predict):
  if (opts.mnb) or (opts.bernoulli) or (opts.sgd):
    while(True):
      input_var = raw_input("Type a tweet, press enter without typing to exit: ")
      if input_var=="":
        break
      X_test = [clearString(input_var)]
      pred = grid_search.predict_proba(X_test)
      print("prediction result {} with probability {}\n".format(target_names[np.argmax(pred)],np.max(pred)))
  else:
    while(True):
      input_var = raw_input("Type a tweet, press enter without typing to exit: ")
      if input_var=="":
        break
      X_test = [clearString(input_var)]
      pred = grid_search.predict(X_test)
      print("prediction result {}\n".format(target_names[pred]))






