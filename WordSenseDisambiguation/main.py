from xml.dom import minidom
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import nltk
import codecs
import sys
import unicodedata
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.corpus import wordnet as wn
import heapq
import math

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def build_vectors(lexelt_item, language, inputfileTrain):
    #''' 
    if language == 'Spanish':
        stopwords = nltk.corpus.stopwords.words('spanish')
        stemmer = SnowballStemmer('spanish')

    if language == 'English':
        stopwords = nltk.corpus.stopwords.words('english')
        stemmer = SnowballStemmer('english')
    #'''

    xmldoc = minidom.parse(inputfileTrain)

    lex_list = xmldoc.getElementsByTagName('lexelt')
    for node in lex_list:
        if lexelt_item == node.getAttribute('item'):
            

            #list holding all words within k of any instance of lexelt 
            s = []

            #list, with each element being a list that holds the words contained in the given instance...kept in order with sense id's list 
            #this info will be basis for context vectors
            sis = []
         
            
            senseids = []
            reftable = []

            inst_list = node.getElementsByTagName('instance')
            for inst in inst_list:
                instance_id = inst.getAttribute('id')
                instance_id = ".".join([replace_accented(instance_id.split('.')[0]), instance_id.split('.', 1)[1]]) 

                sense_id = replace_accented(inst.getElementsByTagName('answer')[0].getAttribute('senseid'))
                if sense_id not in reftable:
                    reftable.append(sense_id)

                sense_id = reftable.index(sense_id)
                if language == 'English':
                    l = inst.getElementsByTagName('context')[0]
                else:
                    l = inst.getElementsByTagName('context')[0].getElementsByTagName('target')[0]
                    
                
                
                frontcontext = nltk.word_tokenize(l.childNodes[0].nodeValue)
                frontcontext.reverse()
                wordsbefore = []
                
                i = 0
                
                for x in range (0, 25):
                    
                    if not language == 'Catalan':  
                        while i < len(frontcontext) and frontcontext[i] in stopwords:
                            i += 1 
                    
                    while i < len(frontcontext) and frontcontext[i] in string.punctuation:
                        i += 1 
                    

                    if i >= len(frontcontext):
                        continue
    
                    wordsbefore.append(frontcontext[i])
                    i += 1
                
                #list to hold 20 words surrounding this instance--represents the S(i) for this instance
                surrwords = []
                for word in wordsbefore:
                    
                    if not language == 'Catalan': 
                        word = stemmer.stem(word)
                    
                    surrwords.append(word)
                   
                    if word in s:
                        continue
                    else:
                        s.append(word)
                    
                    

                if (len(l.childNodes) == 3):
                    backcontext = nltk.word_tokenize(l.childNodes[2].nodeValue)
                    
                    wordsafter = []
                    i = 0
                    
                    for x in range (0, 25):
                        
                        if not language == 'Catalan':
                            while i<len(backcontext) and backcontext[i] in stopwords:
                                i += 1
                        
                        while i<len(backcontext) and backcontext[i] in string.punctuation:
                            i += 1
                        

                        if i>= len(backcontext):
                            continue
                        
                        wordsafter.append(backcontext[i])
                        i += 1
   
                    
                    for word in wordsafter:
                        
                        if not language == 'Catalan':
                             word = stemmer.stem(word)
                        
                        
                        surrwords.append(word)
                        
                        if word in s:
                            continue
                        else:
                            s.append(word)
                    

                sis.append(surrwords)
                senseids.append(sense_id)
                
                
                  
            
            #list, with each element being a list that represents the context vector for a given instance
            #format: [[1, 0, 0, ..., 0], [0, 1, 2, 0, ..., 1], [1, 0, 0, ..., 0]...]
            c_vectors = []
            
            for si in sis:
                vector = []
                for word in s:
                    count = 0
                    for matcher in si:
                        if word == matcher:
                            count += 1
                        else:
                            continue
                    vector.append(count)
                     
                c_vectors.append(vector)

            return (c_vectors, reftable, senseids, s) 
        
 
        else:
            continue
        
    

if __name__ == '__main__':
    
    

    inputfileTrain = sys.argv[1]
    inputfileDev = sys.argv[2]
    outputfile = sys.argv[3]
    language = sys.argv[4]

    #''' 
    if language == 'Spanish':
        stopwords = nltk.corpus.stopwords.words('spanish')
        stemmer = SnowballStemmer("spanish")

    if language == 'English':
        stopwords = nltk.corpus.stopwords.words('english')
        stemmer = SnowballStemmer('english')
    #'''

    xmldoc = minidom.parse(inputfileDev)

    lex_list = xmldoc.getElementsByTagName('lexelt')
    lex_list = sorted(lex_list, key = lambda d: replace_accented(d.getAttribute('item').split('.')[0]))
    
    outfile = codecs.open(outputfile, encoding = 'utf-8', mode = 'w')

    for node in lex_list:
        lexelt_item = node.getAttribute('item')
        vectdata = build_vectors(lexelt_item, language, inputfileTrain)
        lexelt_item = ".".join([replace_accented(lexelt_item.split('.')[0]), lexelt_item.split('.', 1)[1]]) 

        inst_list = node.getElementsByTagName('instance')
        for inst in inst_list:
            instance_id = inst.getAttribute('id')
            instance_id = ".".join([replace_accented(instance_id.split('.')[0]), instance_id.split('.', 1)[1]]) 
            if language == 'English':
                l = inst.getElementsByTagName('context')[0]
            else:
                l = inst.getElementsByTagName('context')[0].getElementsByTagName('target')[0]


            frontcontext = nltk.word_tokenize(l.childNodes[0].nodeValue)
            frontcontext.reverse()
            wordsbefore = []
            i = 0
            
            for x in range (0, 25):
                
                if not language == 'Catalan': 
                    while i < len(frontcontext) and frontcontext[i] in stopwords:
                        i += 1 
                
                while i < len(frontcontext) and frontcontext[i] in string.punctuation:
                   i += 1   
                

                if i >= len(frontcontext):
                    continue
                
                
                if language == 'Catalan':
                    wordsbefore.append(frontcontext[i])
                else: 
                    wordsbefore.append(stemmer.stem(frontcontext[i]))
                
                i += 1
 
            backcontext = nltk.word_tokenize(l.childNodes[2].nodeValue)
            wordsafter = []
            
            i = 0
            
            for x in range (0, 25):
                
                if not language == 'Catalan':
                    while i<len(backcontext) and backcontext[i] in stopwords:
                        i+=1

                while i <len(backcontext) and backcontext[i] in string.punctuation:
                    i+=1

                if i >= len(backcontext):
                    continue
               

                if language == 'Catalan':
                    wordsafter.append(backcontext[i])
                else:
                    wordsafter.append(stemmer.stem(backcontext[i]))
                
                i += 1
            
            context = wordsbefore + wordsafter
            
            c_vector = []
            
            for word in vectdata[3]:
                count = 0
                for matcher in context:
                    if word == matcher:
                        count +=1
                    else:
                        continue
                c_vector.append(count)
            
             
            clf = svm.LinearSVC()
            clf.fit(vectdata[0], vectdata[2])
            outfile.write(lexelt_item + ' ' + instance_id + ' ' + vectdata[1][clf.predict(c_vector)]+ '\n')


