#Imports----------------------------------------------------------------
import time
#from nltk.tag.stanford import StanfordTagger
#from nltk.parse import stanford
import os
import glob
import numpy as np
from numpy import *
from math import *
from sklearn import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale
import string
import nltk.data ,nltk.tag
#from nltk.tag.stanford import StanfordTagger
import math
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier



#Utility Arrays----------------------------------------------------------------

n_authors = 40
n_files = 50
color = []
train = [ []  for i in range(0 ,n_authors)]
test = [ [] for i in range(0,n_authors)]
DICT = [ set() for i in range(0 ,n_authors)]
TP =[0.0 for i in range(0 ,n_authors)]#True positives
FP = [0.0 for i in range(0 ,n_authors) ]#False positives
FN = [0.0 for i in range(0 ,n_authors) ]#False Negatives
t = [[] for i in range(0 ,n_authors * n_files)]
VOCAB = [ {} for i in range(0,n_authors)]
HIT= [ [0 for j in range(0,n_authors)] for i in range(0,n_authors * n_files)]
LTRAIN = []

#Feature Arrays----------------------------------------------------------------

WPS = [0.0 for i in range(0 ,n_authors * n_files)] #words per sentence================== check
WPF = [0.0 for i in range(0 ,n_authors * n_files)] #Words per file==========================check
LD = [0.0 for i in range(0 ,n_authors * n_files)] #Lexical Diversity===========================check
SENTENCES = [0.0 for i in range(0 ,n_authors * n_files)] #No. of sentence========================check
PPS = [0 for i in range(0 ,n_authors * n_files)] #special characters per sentence
QUOTES = [ 0 for i in range(0 ,n_authors * n_files) ] #Quotes per file
CAPWRD = [0 for i in range(0 ,n_authors * n_files)] #Words beginning with caps per file
AWL = [ 0.0 for i in range(0 ,n_authors * n_files) ] #Average word length per file
DIGIT = [0 for i in range(0 ,n_authors * n_files)] #Digits per file
AQL = [0.0 for i in range(0 ,n_authors * n_files)] #Average quotation length
AFL = [0.0 for i in range(0 ,n_authors * n_files)] #Average fragment length
REP = [0.0 for i in range(0 ,n_authors * n_files)] #Repeated words
MAX = [0 for i in range(0 ,n_authors * n_files)]#Author with max no. of common words======================check
SW = [0 for i in range(0 ,n_authors * n_files)]#Short words len <= 4
TRI = [0.0 for i in range(0 ,n_authors * n_files)]#Average Trigram length
TMAX = [0.0 for i in range(0 ,n_authors * n_files)]#Trigram intersection
LTEST = []#No. of lines per file

#File I/O--------------------------------------------------------------------

def readTrain():
    i = 0
    global train
    global LTRAIN
    print 'Reading Training Data.....'
    for x in glob.glob(os.path.join(os.getcwd() + '/C50train','*')):
        if i > 39:
            continue
        for filename in glob.glob( os.path.join(x,'*.txt')):
            f = open(filename,'r')
            s = ""
            for word in f:
                s += word
            train[i].append(s)
            with open(filename) as f:
                LTRAIN.append(sum(1 for _ in f))
        i = i + 1
    print 'Done'

def readTest():
    global test
    global LTEST
    i = 0
    print 'Reading Test Data....'
    for x in glob.glob(os.path.join(os.getcwd() + '/C50test','*')):
        if i > 39:
            continue
        for filename in glob.glob( os.path.join(x,'*.txt')):
            f = open(filename,'r')
            s = ""
            for word in f:
                s += word
            test[i] = test[i].append(s)
            with open(filename) as f:
                LTEST.append(sum(1 for _ in f))
        i = i + 1
    print 'Done'

readTrain()
readTest()
print train[1]

#clf = OneVsRestClassifier(ExtraTreesClassifier(n_estimators=100,max_features = None))
#clf = OneVsRestClassifier(GradientBoostingClassifier())
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=80))
#clf = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=3,algorithm='auto'))
#clf = OneVsRestClassifier(linear_model.SGDClassifier(loss='hinge',penalty='l1'))
#java_path = "C:/Program Files (x86)/Java/jdk1.8.0_25/bin/java.exe"
#os.environ['JAVAHOME'] = java_path
#nltk.internals.config_java("C:/Program Files (x86)/Java/jdk1.8.0_25/bin/java.exe")
#path_to_model = "E:/Minor Project/stanford-postagger-full-2015-01-30/models/english-bidirectional-distsim.tagger"
#path_to_jar = "E:/Minor Project/stanford-postagger-full-2015-01-30/stanford-postagger.jar"
#tagger = nltk.tag.stanford.POSTagger(path_to_model, path_to_jar)


#Model Fitting----------------------------------------------------------
def model(mode = 0):
    if mode == 0:
        print "Analyzing vocabularies....."
        for i in range(0 ,n_authors):
            for j in range(0 ,n_files):
                tokens = nltk.word_tokenize( train[i][j])
                DICT[i] = set.union(DICT[i] , set(tokens))
    for i in range(0 ,n_authors):
        if mode == 0:
            print "Training Author " + str(i) + "...."
        else:
            print "Extracting Author " + str(i) + "...."
        for j in range(0 ,n_files):
            f = 0.0
            p = 0.0
            q = 0.0
            m1 = 0
            aql = 0.0
            aqs = 0.0
            d = 0.0
            u = 0.0
            if mode == 0:
                tokens = nltk.word_tokenize( train[i][j])
                S = train[i][j]
            else:
                tokens = nltk.word_tokenize( test[i][j])
                S = test[i][j]
            LD[i * n_files + j] = float(len(set(tokens)))
            
            WPF[i * n_files + j] = float(len(tokens))

            REP[ i * n_files + j ] = WPF[i * n_files + j] - LD[i * n_files + j]

            TRI[ i * n_files + j ] = 0.0
            x = 0.0
            c = 0.0
            grams = nltk.trigrams(tokens)
            for k in range(0 ,n_authors):
                    HIT[i * n_files + j][k] = 0.0
            for obj in grams:
                TRI[ i * n_files + j ] = TRI[ i * n_files + j ] + float(len(obj[0]) + len(obj[1]) + len(obj[2]))
                for k in range(0 ,n_authors):
                    if obj[0] in DICT[k]:
                        HIT[i * n_files + j][k] += 1
                    if obj[1] in DICT[k]:
                        HIT[i * n_files + j][k] += 1
                    if obj[2] in DICT[k]:
                        HIT[i * n_files + j][k] += 1
                x = x + 1
            if x != 0:
                TRI[ i * n_files + j ] /= x

            m = 0
            v = HIT[i * n_files + j][m]
            for k in range(1 ,n_authors):
                if HIT[i * n_files + j][k] > v:
                    v = HIT[i * n_files + j][k]
                    m = k
            TMAX[i * n_files + j] = pow(3 ,m)

            #PREP[ i * n_files + j ] = 0
            #TAG = tagger.tag(tokens)
            #print time.strftime("now = %X")
            #for item in TAG:
             #  if item[1] == 'P':
              #      PREP[ i * n_files + j ] = PREP[ i * n_files + j ] + 1
            #print time.strftime("end = %X")
            for ch in S:
                if ch == '.':
                    f = f + 1
                if ch == ',':
                    f = f + 1
                if ch in string.punctuation:
                    p = p + 1
                if ch == '"':
                    q = q + 1
                    m1 = (m1 + 1) % 2
                    if m1 % 2 == 0:
                        aql += aqs
                        aqs = 0.0
                if ch >= 48 and ch <= 57:
                    d = d + 1
                if m1 % 2 == 1:
                    aqs = aqs + 1
                if ch >= 65 and ch <= 90:
                    u = u + 1
            QUOTES[  i * n_files + j ] = q / 2

            DIGIT[i * n_files + j] = d

            CAPWRD[i * n_files + j] = u
            
            if f == 0:
                AFL[i * n_files + j] = 0.0
            else:
                AFL[i * n_files + j] = float(len(S))/ f
            
            if q == 0:
                AQL[i * n_files + j] = 0.0
            else:
                AQL[i * n_files + j] = aql / (q / 2)
            
            x = 0.0
            for word in tokens:
                x = x + float(len(word))
            AWL[  i * n_files + j ] = x / WPF[ i * n_files + j ]

            for k in range(0 ,n_authors):
                HIT[i * n_files + j][k] = len( set.intersection(DICT[k] ,set(tokens)) )
            m = 0
            v = HIT[i * n_files + j][m]
            for k in range(1 ,n_authors):
                if HIT[i * n_files + j][k] > v:
                    v = HIT[i * n_files + j][k]
                    m = k
            MAX[i * n_files + j] = pow(4 ,m)
            
            if mode == 0:
                tokens = nltk.sent_tokenize( train[i][j])
            else:
                tokens = nltk.sent_tokenize( test[i][j])
            SENTENCES[i * n_files + j] = float(len(tokens))

            PPS[i * n_files + j] = p / SENTENCES[i * n_files + j]
            
            x = 0.0
            if SENTENCES[i * n_files + j] != 0:
                x = WPF[i * n_files + j] / SENTENCES[i * n_files + j]
            WPS[i * n_files + j] = x
#-------------------------SCALING WEIGHTS------------------------------------
            AWL[i * n_files + j] = pow(AWL[i * n_files + j] , 3)

            #TRI[i * n_files + j] = pow(TRI[i * n_files + j] ,3)
#-------------------------------------------------------------------
            #print "file " + str( i * n_files + j )
    #print WPS
    print "Fitting data..."
    y = []
    L = []
    L.append(WPS[0])
    L.append(WPF[0])
    L.append(LD[0])
    L.append(SENTENCES[0])
    L.append(PPS[0])
    L.append(QUOTES[0])
    L.append(CAPWRD[0])
    L.append(AWL[0])
    L.append(DIGIT[0])
    L.append(AQL[0])
    L.append(AFL[0])
    L.append(REP[0])
    L.append(MAX[0])
    L.append(TRI[0])
    L.append(TMAX[0])
    #L.append(WMAX[0])
    #L.append(PREP[0])
    if mode == 0:
        L.append(LTRAIN[0])
        t[0].append(WPS[0])
        t[0].append(WPF[0])
        t[0].append(LD[0])
        t[0].append(SENTENCES[0])
        t[0].append(PPS[0])
        t[0].append(QUOTES[0])
        t[0].append(CAPWRD[0])
        t[0].append(AWL[0])
        t[0].append(DIGIT[0])
        t[0].append(AQL[0])
        t[0].append(AFL[0])
        t[0].append(REP[0])
        t[0].append(MAX[0])
        t[0].append(TRI[0])
        t[0].append(LTRAIN[0])
    else:
        L.append(LTEST[0])
    X = array(L)
    for  i in range(1 ,n_files * n_authors):
        L = []
        L.append(WPS[i])
        L.append(WPF[i])
        L.append(LD[i])
        L.append(SENTENCES[i])
        L.append(PPS[i])
        L.append(QUOTES[i])
        L.append(CAPWRD[i])
        L.append(AWL[i])
        L.append(DIGIT[i])
        L.append(AQL[i])
        L.append(AFL[i])
        L.append(REP[i])
        L.append(MAX[i])
        L.append(TRI[i])
        L.append(TMAX[i])
        #L.append(WMAX[i])
        #L.append(PREP[i])
        if mode == 0:
            L.append(LTRAIN[i])
            t[i].append(WPS[i])
            t[i].append(WPF[i])
            t[i].append(LD[i])
            t[i].append(SENTENCES[i])
            t[i].append(PPS[i])
            t[i].append(QUOTES[i])
            t[i].append(CAPWRD[i])
            t[i].append(AWL[i])
            t[i].append(DIGIT[i])
            t[i].append(AQL[i])
            t[i].append(AFL[i])
            t[i].append(REP[i])
            t[i].append(MAX[i])
            t[i].append(TRI[i])
            t[i].append(LTRAIN[i])
        else:
            L.append(LTEST[i])
        a = array(L)
        X = np.vstack((X,a))
    for i in range(0,n_authors):
        for j in range(0 ,n_files):
            y.append(i)
    Y = array(y)
    #X = scale(X)
    if mode == 0:
        clf.fit(X ,Y)
    R = clf.predict(X)
    plt.scatter( WPS , LD ,c = [ (int)(R[i] % 40) for i in range(0 ,2000)],s = 100 )
    plt.show()
    c = 0.0
    for i in range(0 ,n_authors * n_files):
        if R[i] == Y[i]:
            c = c + 1
    print str(c) + " / " + str(n_authors * n_files) + " = " + str(c * 100/ (n_authors * n_files)) + " % "
    if mode == 1:
        for i in range(0 ,n_authors):
            for j in range(0 ,n_files):
                if R[i * n_files + j] == Y[i* n_files + j]:
                    TP[R[i* n_files + j]] = TP[R[i * n_files + j]] + 1
                else:
                    FP[R[i * n_files + j]] = FP[R[i * n_files + j]] + 1
                    FN[Y[i* n_files + j]] += 1
        c = 0.0
        recall = 0.0
        for i in range(0 ,n_authors):
            if TP[i] + FP[i] != 0:
                c += TP[i] / (TP[i] + FP[i])
                recall += TP[i] / (TP[i] + FN[i])
        c /= n_authors
        recall /= n_authors
        print "Precison " + " = " + str(c * 100) + " % "
        print "Recall " + " = " + str(recall * 100) + " % "
        print "Accuracy = " + str((2 * c * recall  * 100) / (c + recall)) + "%"
model(0)
model(1)

