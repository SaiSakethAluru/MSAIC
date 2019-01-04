import math
import pickle
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import string

#Initialize Global variables 
docIDFDict = {}
avgDocLength = 0

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    for line in f:
        # print(line)
        passage = line.strip().lower().split("\t")
        # print(passage)
        passage = passage[2]
        # passage = word_tokenize(line)
        # tags = pos_tag(passage)
        # print(len(passage))
        # for i in range(len(passage)):
            # p_tag = get_wordnet_pos(tags[i][1])
            # if(p_tag!=''):
                # p = lemmatizer.lemmatize(passage[i],pos=p_tag)
            # else:
                # p = stemmer.stem(passage[i])
        fw.write(passage+"\n")
        # token = pos_tag(passage)
        # passage = lemmatizer.lemmatize(passage,pos=token)

        # fw.write(passage+"\n")
    f.close()
    fw.close()



# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula 
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :

    global docIDFDict,avgDocLength
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0

    for line in open(corpusfile,"r",encoding="utf-8") :
        # doc = line.strip().split(delimiter)
        doc = word_tokenize(line)
        totalDocLength += len(doc)
        doc = list(set(doc)) # Take all unique words
        stop_words = set(stopwords.words('english'))
        doc = [w for w in doc if w not in stop_words or w not in list(string.punctuation)]
        tags = pos_tag(doc)
        # for word in doc : #Updates n(q_i) values for all the words(q_i)
        #     if word not in docFrequencyDict :
        #         docFrequencyDict[word] = 0
        #     docFrequencyDict[word] += 1
        for i in range(len(doc)):
            p_tag = get_wordnet_pos(tags[i][1])
            if(p_tag!=''):
                word = lemmatizer.lemmatize(doc[i],pos=p_tag)
            else:
                word = stemmer.stem(doc[i])
            if word not in docFrequencyDict:
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1
        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%5000==0):
            print(numOfDocuments)                

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        # print(numOfDocuments)
        # print(docFrequencyDict[word])
        idf_ratio = (numOfDocuments) / (docFrequencyDict[word])
        if idf_ratio>0:
            docIDFDict[word] = math.log(idf_ratio, base) #Why are you considering "numOfDocuments - docFrequencyDict[word]" instead of just "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments

     
    pickle_out = open("docIDFDict.pickle","wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()


    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)



#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength
    stop_words = set(stopwords.words('english'))
    # query_words= Query.strip().lower().split(delimiter)
    query_words = word_tokenize(Query.lower())
    # passage_words = Passage.strip().lower().split(delimiter)
    passage_words = word_tokenize(Passage.lower())
    query_words = [w for w in query_words if w not in stop_words or w not in list(string.punctuation)]
    passage_words = [w for w in passage_words if w not in stop_words or w not in list(string.punctuation)]
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        # tokens = line.strip().lower().split("\t")
        tokens_ = word_tokenize(line.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens_ ]
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage) 
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = tokens[0]
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%5000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()


if __name__ == '__main__' :

    # inputFileName = "traindata.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
    inputFileName = "full_model/Data.tsv"
    # testFileName = "data_split.tsv"  # This file should be in the following format : queryid \t query \t passage \t passageid # order of the query
    testFileName = "full_model/eval1_unlabelled.tsv"
    corpusFileName = "corpus.tsv" 
    # outputFileName = "answer_temp.tsv"
    outputFileName = "answer.tsv"

    GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
    print("Corpus File is created.")
    IDF_Generator(corpusFileName)   # Calculates IDF scores. 
    #RunBM25OnTestData(testFileName,outputFileName)
    print("IDF Dictionary Generated.")
    RunBM25OnEvaluationSet(testFileName,outputFileName)
    print("Submission file created. ")

