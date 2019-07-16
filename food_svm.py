# -*- coding: utf-8 -*-import twitter
import twitter
import json
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from datetime import datetime

totPredTwtCnt = 0
posPredTwtCnt = 0
negPredTwtCnt = 0

maxFeatureFreq = 0
vocab1 = dict()
consumer_api_key = 'LCM9O3vbTvHsS6YS966OHKqcA'
consumer_api_secret = 'FDgrW27FQRRZSEDFWfd2J5lZAVhafGtkVZwINvQG5ZdVIQ1I62'
access_token_key = '1098880175918497793-zyCsrnUwrZhgLSEpJ7lVscEeR6IwOS'
access_token_secret = '61BqfKnBpg7InJbRrsnveZShxCNCPQnbTLPIjuKN9Svl4'

myApi = twitter.Api(consumer_key=consumer_api_key, consumer_secret=consumer_api_secret,
                    access_token_key=access_token_key, access_token_secret=access_token_secret)

out_file = ".\AllTweets.txt";
tweetFile = open(out_file, "w")

prj_answer_file = open(".\prj_answer_file.txt", "w")

predicted_tweets_file = open(".\predicted_tweets.txt", "w")

def get_tweets():
    query = 'Chipotle'
    MAX_ID = None
    for it in range(5): # Retrieve up to 200 tweets
        tweets = [json.loads(str(raw_tweet)) for raw_tweet
                 in myApi.GetSearch(term=query, count=1000, max_id=MAX_ID)]
        if tweets:
            for tweet in tweets:
                MAX_ID = tweets[-1]['id']
                tweetText = tweet['text']
                print 'User Tweet >> '
                print MAX_ID, tweetText
                tweetFile.write((tweetText.encode('utf-8')) + "\n")

def find_features():
    stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
                 "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst",
                 "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
                 "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been",
                 "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill",
                 "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
                 "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
                 "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
                 "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first",
                 "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further",
                 "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
                 "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
                 "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep",
                 "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile",
                 "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
                 "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
                 "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
                 "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per",
                 "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
                 "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so",
                 "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such",
                 "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence",
                 "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv",
                 "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
                 "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up",
                 "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence",
                 "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
                 "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
                 "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the",
                 "...", "https://t.co/ggwgq2ug1h", "@jamesakajim:", "๐’", "just", "brotha", "forgot",
                 "๐ด๐‘๐•ต๐พโ€โ๏ธโ๐พ๐", "chipotle...", "chipotle", 'https://t.co/axlgxjms3d',
                 "https://t.co/fu1dlpibei", "@thejose8a:", "@ayye_pap:", "benjamin", "iโ€m", "woke" ]

    tweets = []
    for line in open('labeled tweets.txt').readlines()[:1000]:
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
    vocab = {term: freq for term, freq in vocab.items() if freq > 21}
    global maxFeatureFreq
    for kword in vocab:
        if (vocab.get(kword) > maxFeatureFreq):
            maxFeatureFreq = vocab.get(kword)
        print("feature: %s with frequency: %s" % (kword, vocab.get(kword)))
        prj_answer_file.write("feature: %s with frequency: %s \n" % (kword, vocab.get(kword)))

    # Generate an id (starting from 0) for each term in vocab
    global vocab1
    vocab1 = vocab
    vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}

    print maxFeatureFreq
    print "The number of keywords used for generating features (frequencies): ", len(vocab)
    prj_answer_file.write("The number of keywords used for generating features (frequencies): %s \n" % len(vocab))

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

    print "The total number of training tweets: {} ({} positives, {}: negatives)".format(len(y), sum(y),
                                                                                         len(y) - sum(y))
    prj_answer_file.write("The total number of training tweets: {} ({} positives, {}: negatives) \n".format(len(y), sum(y),
                                                                                        len(y) - sum(y)))

    a = np.array(X)
    b = np.array(y)

    print(a)
    prj_answer_file.write("<<2-Dimensional ndarray X >> \n")
    for aitem in a:
        prj_answer_file.write('%s\n' % aitem)
    print(b)
    prj_answer_file.write("<<2-Dimensional ndarray y \n>>")
    for bitem in b:
        prj_answer_file.write('%s\n' % bitem)

    # 10 folder cross validation to estimate the best w and b
    svc = svm.SVC(kernel='linear')
    Cs = range(1, 20)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X, y)

    print "The estimated w: "
    print clf.best_estimator_.coef_
    prj_answer_file.write("The estimated w: %s\n" %clf.best_estimator_.coef_)

    print "The estimated b: "
    print clf.best_estimator_.intercept_
    prj_answer_file.write("The estimated b: %s\n" % clf.best_estimator_.intercept_)

    print "The estimated C after the grid search for 10 fold cross validation: "
    print clf.best_params_
    prj_answer_file.write("The estimated C after the grid search for 10 fold cross validation: %s\n" %clf.best_params_)

    # predict the class labels of new tweets
    tweets = []
    for line in open('unlabeled tweets.txt').readlines():
        tweets.append(line)

    # Generate X for testing tweets
    test_X = []
    for text in tweets:
        x = [0] * len(vocab)
        print "x", x
        terms = [term for term in text.split() if len(term) > 2]
        print "terms", terms
        for term in terms:
            if vocab.has_key(term):
                x[vocab[term]] += 1
        test_X.append(x)
    test_y = clf.predict(test_X)
    print "test_y", test_y

    for i in range(len(test_y)):
        pred_result, tweetText = (test_y[i], tweets[i])
        #print "pred_result", pred_result, "tweetText", tweetText
        predicted_tweets_file.write("%s, %s" %(pred_result, tweetText))

    global totPredTwtCnt
    global posPredTwtCnt
    global negPredTwtCnt
    totPredTwtCnt = len(test_y)
    posPredTwtCnt = sum(test_y)
    negPredTwtCnt = len(test_y) - sum(test_y)
    print "The total number of testing tweets: {} ({} are predicted as positives, " \
          "{} are predicted as negatives)".format(len(test_y), sum(test_y), len(test_y) - sum(test_y))
    print "The total number of testing tweets: {} ({} are predicted as positives, " \
          "{} are predicted as negatives)".format(totPredTwtCnt, posPredTwtCnt, negPredTwtCnt)

def plotResult(posPredTwtCntLocal, negPredTwtCntLocal, topic):
    print "The total number of testing tweets: {} ({} are predicted as positives, " \
          "{} are predicted as negatives)".format(totPredTwtCnt, posPredTwtCnt, negPredTwtCnt)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Positive', 'Negative'
    sizes = [posPredTwtCntLocal, negPredTwtCntLocal]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    print "explode1"
    fig1, ax1 = plt.subplots()
    print "fig1, ax1 = plt.subplots()>>>>>>>>>"
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    print "showing"

    #curr_time = datetime.now()
    #formatted_time = curr_time.strftime('%m%d%Y_%H_%M_%S_%f')
    #print(formatted_time)

    #plt.savefig('pie'+formatted_time)
    plt.savefig('static\\pie_' + topic)
    plt.show()
    #plt.clf()
    #plt.close('all')

def plotResultBar(vocabLocal, topic):
    plt.style.use('ggplot')
    xFeatureWord = []
    yFeatureFreq = []

    for keyword in vocabLocal:
        xFeatureWord.append(keyword)
        yFeatureFreq.append(vocabLocal.get(keyword))
    #x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
    #energy = [5, 6, 15, 22, 24, 8]

    x_pos = [i for i, _ in enumerate(xFeatureWord)]

    plt.bar(x_pos, yFeatureFreq, color='green', width=0.95, )
    plt.xlabel("Feature Word")
    plt.ylabel("Frequency")
    plt.title("Feature Frequency")

    plt.xticks(x_pos, xFeatureWord, rotation='vertical')
    #plt.margins(0.2)
    plt.autoscale(enable=True, axis='both', tight=None)
    #curr_time = datetime.now()
    #formatted_time = curr_time.strftime('%m%d%Y_%H_%M_%S_%f')
    #plt.savefig('bar' + formatted_time)
    plt.savefig('static\\bar_' + topic)
    plt.show()

#get_tweets()
#tweetFile.close()
find_features()
prj_answer_file.close()
predicted_tweets_file.close()
plotResult(posPredTwtCnt, negPredTwtCnt, 'food')
plotResultBar(vocab1, 'food')

def displayGUI():
    global out_file, tweetFile, prj_answer_file, predicted_tweets_file
    out_file = ".\AllTweets.txt";
    tweetFile = open(out_file, "w")

    prj_answer_file = open(".\prj_answer_file.txt", "w")

    predicted_tweets_file = open(".\predicted_tweets.txt", "w")

    find_features()
    prj_answer_file.close()
    predicted_tweets_file.close()
    plotResult(posPredTwtCnt, negPredTwtCnt, 'food')
    plotResultBar(vocab1, 'food')
    return