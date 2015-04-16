import sys
import nltk
import math

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    wordcount = {}
    

    #go thru sentence by sentence, word by word, updating wordcount dictionary
    for sentence in wbrown:
        for word in sentence:
            if word in wordcount:
                wordcount[word] += 1
            else:
                wordcount[word] = 1

    

    knownwords = []
    
    #if a given word has more than 5 occurrances, add it to knownwords
    for key,value in wordcount.items():
        if value > 5:
            knownwords.append(key) 
    

    

    return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    
    rare = []
   
    #convert to frozen set for faster lookup
    knownwordsSet = frozenset(knownwords)
       
    for sentence in brown:
        newsentence = []
        for word in sentence:
            if word in knownwordsSet:
                newsentence.append(word)
            else:
                newsentence.append('_RARE_')

        rare.append(newsentence)
     
    

    return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')

    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    
    outfile.close()


#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
    trigram_count = {}
    bigram_count = {}
    
    for sentence in tbrown:

        #get bigram counts
        bigram_tuples = tuple(nltk.bigrams(sentence))
        for bigram in bigram_tuples:
            if bigram in bigram_count:
                bigram_count[bigram] += 1
            else:
                bigram_count[bigram] = 1
         
        #get trigram counts
        trigram_tuples = tuple(nltk.trigrams(sentence))
        for trigram in trigram_tuples:
            if trigram in trigram_count:
                trigram_count[trigram] += 1
            else:
                trigram_count[trigram] = 1
         
    qvalues = {}
    
    #take the trigram count and divide it by bigram count of latter two tages
    for trigram, count in trigram_count.items():
        qvalues[trigram] = math.log(float(count)/bigram_count[(trigram[0], trigram[1])], 2)

 
    return qvalues

#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output 
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word 
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    

    evalues = {}
    taglist = []
    tag_count = {}
    wordtag_count = {}

    for sentence in tbrown:
        for tag in sentence:
            if tag in tag_count:
                tag_count[tag] +=1
            else:
                tag_count[tag] = 1
        
    #zip through lists to create merged list
    matchsentencelist = zip(wbrown, tbrown)
    for matchsentence in matchsentencelist:
        #merge the words in the individual sentence
        wordtaglist = zip (matchsentence[0], matchsentence[1])
        
        #calculate word/tag emissions
        for wordtag in wordtaglist:
            if wordtag in evalues:
                evalues[wordtag] += 1
            else:
                evalues[wordtag] = 1
    
    for wordtag, value in evalues.items():
        
        #get log probability for each word/tag
        evalues[wordtag] = math.log((float(value) / tag_count[wordtag[1]]), 2)

    #add each tag to taglist
    for tag in tag_count:
        taglist.append(tag)
     
    
    
    return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues: 
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords), 
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a 
#sentence tagged in the WORD/TAG format 
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams
#evalues is from the return of calc_emissions() 
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    brown_smoothed = replace_rare(brown, knownwords)
    
    sentenceTags = []
    tagged = []
        

    i=0
    
    for sentence in brown_smoothed:
        
        #original sentence, without RARE replacement..will be used as return result
        osentence = brown[i]
        i +=1

        #dictionary to hold viterbi values of 
        #keys are a 2 tuple (k,(u, v)) where k is word#, (u,v) is tag touple  
        #values are a 2 tuple (probability, backpointer)
        vittable = {}

        #initialize k=0 values for table
        for tag1 in taglist:
            for tag2 in taglist:
                if tag1 == tag2 == '*':
                    vittable[-1, (tag1, tag2)] = (0, None)
                else:
                    vittable[-1, (tag1, tag2)] = (-1000, None)
                
        n = len(sentence)

        #loop through sentence to fill in table
        for k in range(0, n):
            for u in taglist:
                for v in taglist:
                    
                    #initialize 'pi' probability to very low number, so any decent tag for w is better 
                    pi = -100000
                    bp = None
                    for w in taglist:
                        piTemp = vittable[k-1, (w, u)][0] + qvalues.get((w, u, v), -1000) + evalues.get((sentence[k],v), -1000)
                        if piTemp > pi:
                            pi = piTemp
                            bp = w
                    vittable[k, (u, v)] = (pi, bp)
                             
        thisSentenceTags = []
         
        #find best trigram ending in STOP, follow backpointers
        #the best trigram info is stored in winner
        finalprob = -100000
        winner = None
        for u in taglist:
            for v in taglist:
                
                finalprobTemp = vittable[n-1, (u, v)][0] + qvalues.get(('STOP', v, u), -1000)
                if finalprobTemp > finalprob:
                    
                    finalprob = finalprobTemp
                    winner = (n-1, (u, v))
        
        #put v into the sentence tags
        thisSentenceTags.append(winner[1][1])
       
        #follow backpointers and put taggs into thisSentenceTags
        for k in range(n-2, -1, -1):
            thisSentenceTags.append(winner[1][0])
            winner = (k, (vittable[winner][1], winner[1][0]))
            
        #reverse the list, because we put in the last tag in first    
        thisSentenceTags.reverse()
        
        #sentence needs to be a string
        taggedSentence = ""

        #use osentence, because we don't want rare tags
        for x in range (0, n):
            taggedSentence += osentence[x] + '/' + thisSentenceTags[x] + ' '
        
        taggedSentence += '\n'
        
        tagged.append(taggedSentence)
        
        
    return tagged

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of tagged sentences. Each sentence is in the WORD/TAG format and is a string rather than a list of tokens.
def nltk_tagger(brown):
    tagged = []
    from nltk.corpus import brown as brownimport
    training=brownimport.tagged_sents(tagset='universal')
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)

    for sentence in brown:
        taggedSentence = trigram_tagger.tag(sentence)
        sentencetokens = []
        for wordtag in taggedSentence:
            wordtagstring = '/'.join(wordtag)
            sentencetokens.append(wordtagstring) 
        tagged.append(sentencetokens)

    return tagged

def q6_output(tagged):
    outfile = open('B6.txt', 'w')
    for sentence in tagged:
        output = ' '.join(sentence) + '\n' 
        outfile.write(output)
    outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only) 
def split_wordtags(brown_train):
                   
    wbrown = []
    tbrown = []
   
    for sentence in brown_train:
        words = sentence.split()
        words.insert(0, '*/*')
        words.insert(0, '*/*')
        words.append('STOP/STOP')
        wordList = []
        tagList = []
        for word in words:
            splitword = word.rsplit("/", 1)
            wordList.append(splitword[0])
            tagList.append(splitword[1])
        
        wbrown.append(wordList)
        tbrown.append(tagList)

    return wbrown, tbrown

def main():
    #open Brown training data
    infile = open("Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    #split words and tags, and add start and stop symbols (question 1)
    wbrown, tbrown = split_wordtags(brown_train)
           
    #calculate trigram probabilities (question 2)
    qvalues = calc_trigrams(tbrown)
    
    #question 2 output
    q2_output(qvalues)

    #calculate list of words with count > 5 (question 3)
    knownwords = calc_known(wbrown)

    #get a version of wbrown with rare words replace with '_RARE_' (question 3)
    wbrown_rare = replace_rare(wbrown, knownwords)

    #question 3 output
    q3_output(wbrown_rare)

    #calculate emission probabilities (question 4)
    evalues, taglist = calc_emission(wbrown_rare, tbrown)

    #question 4 output
    q4_output(evalues)

    #delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare

    #open Brown development data (question 5)
    infile = open("Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()
    
    #tokenize data
    for x in range (0, len(brown_dev)):
        brown_dev[x] = brown_dev[x].split()
        


    #replace rare words in brown_dev (question 5)
    brown_dev_rare = replace_rare(brown_dev, knownwords)

    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)

    #question 5 output
    q5_output(viterbi_tagged)

    #do nltk tagging here
    nltk_tagged = nltk_tagger(brown_dev)
    
    #question 6 output
    q6_output(nltk_tagged)
if __name__ == "__main__": main()
