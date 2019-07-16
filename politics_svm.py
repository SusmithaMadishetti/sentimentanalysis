from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, GridSearchCV

from food_svm import plotResult
from food_svm import plotResultBar

totPredTwtCnt = 0
posPredTwtCnt = 0
negPredTwtCnt = 0

maxFeatureFreq = 0
global vocab1

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

tweets = []
for line in open('training.txt').readlines()[:1000]:
    items = line.split(',')
    tweets.append([int(items[0]), items[1].lower().strip()])

# Extract the vocabulary of keywords
vocab = dict()
for class_label, text in tweets:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if vocab.has_key(term):
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1

# Remove terms whose frequencies are less than a threshold (e.g., 10)
vocab = {term: freq for term, freq in vocab.items() if freq > 15}
global maxFeatureFreq
for kword in vocab:
    if (vocab.get(kword) > maxFeatureFreq):
        maxFeatureFreq = vocab.get(kword)
    print("feature: %s with frequency: %s" % (kword, vocab.get(kword)))
    #prj_answer_file.write("feature: %s with frequency: %s \n" % (kword, vocab.get(kword)))
# Generate an id (starting from 0) for each term in vocab
vocab1 = vocab
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
print "The number of keywords used for generating features (frequencies): ", len(vocab)

# Generate X and y
X = []
y = []
for class_label, text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split()]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    y.append(class_label)
    X.append(x)

totPredTwtCnt = len(y)
posPredTwtCnt = sum(y)
negPredTwtCnt = len(y) - sum(y)
print "The total number of training tweets: {} ({} positives, {}: negatives)".format(len(y), sum(y), len(y) - sum(y))
print "The total number of training tweets: {} ({} positives, {}: negatives)".format(totPredTwtCnt, posPredTwtCnt, negPredTwtCnt)
plotResult(posPredTwtCnt, negPredTwtCnt, 'politics')
# 10 folder cross validation to estimate the best w and b
svc = svm.SVC(kernel='linear')
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X, y)

print "The estimated w: "
print clf.best_estimator_.coef_

print "The estimated b: "
print clf.best_estimator_.intercept_

print "The estimated C after the grid search for 10 fold cross validation: "
print clf.best_params_

# predict the class labels of new tweets
tweets = []
for line in open('testing.txt').readlines():
    tweets.append(line)

# Generate X for testing tweets
test_X = []
for text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    test_X.append(x)
test_y = clf.predict(test_X)

print "The total number of testing tweets: {} ({} are predicted as positives, {} are predicted as negatives)".format(len(test_y), sum(test_y), len(test_y) - sum(test_y))

print "\n",clf.fit(X,y)
print "\n",clf.predict(X)

# print 100 example tweets and their class labels

print sum(y), len(y)
with open ('politics_predicted_tweets.txt','w') as answer_result:
    idx=0
    for tweet in tweets:
        answer_result.write(str(y[idx])+ " "+tweet)
        idx=idx+1
# print 100 example tweets and their class labels
# for idx in range(1,100):
#     print 'Sentiment Class (1 means positive; 0 means negative): ', y[idx]
#     print 'TEXT: ', idx, tweets[idx]
#
# print sum(y), len(y)
plotResultBar(vocab1, 'politics')

