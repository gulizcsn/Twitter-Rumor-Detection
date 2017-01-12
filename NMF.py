from __future__ import print_function
from nltk.corpus import stopwords
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import os 
import nltk
import pickle
n_samples = 2000000
n_features = 10000
n_topics = 20
n_top_words = 20



stopset = set(stopwords.words('english'))

#to expand stopset, comment out the below update command

#stopset.update(("retweet","rt","rts","women","men","000","thank","thanks","tweet","tweets","let","get","got","think","vote","lot","lots","say","says","know","guys","guy","voto","votes","today","lol",\
#	"tonight","week","people","true","make","need","good","day","win","look","looks","really","thing","said","say","says","tell","told","tells","voting","days","want","voting","making","follow","unfollow",\
#	"way","man","news","come","twitter","yes","real","big","best","girl","right","person","dont","adn","bday","wish","right","poll","video","hope","life","soon","wait","gon","na","going","fav",\
#	"ago","years","make"))


#tokenize the words
def nltk_tokenize():

	myNames =[]
	# if already tokenized, return it. or the file in removed_dublicates_en.txt will be tokenized
	if os.path.isfile('tokens.txt'):
		with open("tokens.txt", "rb") as fp:   # Unpickling
			myNames = pickle.load(fp)
			return myNames


	with open('removed_dublicates_en.txt', 'r') as f:
		
		for line in f:

			# 	return myNames
			if len(line.strip())>0:
				tokens = nltk.tokenize.word_tokenize(line)
				myNames.append(" ".join([w for w in tokens if not w in stopset]))


	with open("tokens.txt", "wb") as fp:   
		pickle.dump(myNames, fp)

	return myNames


# function to print out first 100 sentences  after being tokenized 
def printTokenizeWith(tokens,file):
	counter = 0
	with open(file, 'r') as f:
		
		for line in f:
			if(counter==100):
				return
			print(line,end="")
			print(tokens[counter])
			print()
			counter+=1



#print out the most significant words
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



print("Loading dataset...")
t0 = time()


tokens = nltk_tokenize()
#print(tokens)

data_samples = tokens[:n_samples]
#print(data_sample[:10])
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.1, min_df=20,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.01, min_df=20,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.3, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)








"""
def tokenize(input,output):

	f = open(output,'w')

	with open(input) as inFile:
	    for line in inFile:
   		    seen = set()
	        if not line in seen:
	        	f.write(line)
	        	seen.add(line)


	f.close()

tokenize('names.txt','tokenized.txt')	
"""

#print(nltk_tokenize())

#for sentence in tokens:
#	print(sentence)

#printTokenizeWith(tokens,'removed_dublicates_en.txt')

