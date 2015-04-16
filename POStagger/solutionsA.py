import nltk
import math

#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
def calc_probabilities(brown):
    
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    
    
    #calculate aggregate unigram counts sentence by sentence
    for sentence in brown:
        tokens = nltk.word_tokenize(sentence)
        tokens.insert(0,'*')
        tokens.insert(0, '*')
        tokens.append('STOP')
        
        #update unigram counts
        for unigram in tokens:
            unigram = (unigram,)
            if unigram in unigram_count:
                unigram_count[unigram] += 1
            else:
               unigram_count[unigram] = 1
        
        #update bigram counts
        bigram_tuples = tuple(nltk.bigrams(tokens))
        for bigram in bigram_tuples:
            if bigram in bigram_count:
                bigram_count[bigram] += 1
            else:
                bigram_count[bigram] = 1
        
        #update trigram counts
        trigram_tuples = tuple(nltk.trigrams(tokens))
        for trigram in trigram_tuples:
            if trigram in trigram_count:
                trigram_count[trigram] += 1
            else:
                trigram_count[trigram] = 1
    
    #*'s aren't figured into our unigram model so we eliminate them from unigram count
    totaluni = sum(unigram_count.values()) - unigram_count[('*',)]
   
    #divide unigram_count of * by 2 because we put two *'s for each sentence
    #and as this is used for bigram model, which only should have one * at sentence beginning
    unigram_count[('*',)] = unigram_count[('*'),]/2

    #calculate unigram probabilities (excluding the unigram *)
    for unigram, count in unigram_count.items():
        if unigram == ('*',):
            continue
        else:
            unigram_p[unigram] = math.log((float(count)/totaluni), 2)
   
    #calculate bigram probabilities (excluding the bigram *, *)
    for bigram, count in bigram_count.items():
        if bigram == ('*', '*'):
            continue
        else:
            bigram_p[bigram] = math.log(float(count)/unigram_count[(bigram[0],)], 2)
           #testing: print bigram, ' count: ', count, ' unigram_count: ', unigram_count[(bigram[0],)]
    
    #calculate trigram probabilities
    for trigram, count in trigram_count.items():
        trigram_p[trigram] = math.log(float(count)/bigram_count[(trigram[0], trigram[1])], 2)

    return unigram_p, bigram_p, trigram_p

#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()
    
#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):
    scores = []
    for sentence in data:
        tokens = nltk.word_tokenize(sentence)
        score = 0
        
        #unigram sentence probabilities: only append STOP to end of each sentence
        if (n == 1):
            tokens.append('STOP')

            #score is simply the sum of the log probabilities of each word in sentence 
            for unigram in tokens:
                score += ngram_p[(unigram,)]
        
        #bigram sentence probabilities: need one start symbol and STOP symbol at end
        if (n==2):
            tokens.insert(0, '*')
            tokens.append('STOP')

            #split sentence into bigrams, sum the log probabilities of each bigram 
            bigram_tuples = tuple(nltk.bigrams(tokens))
            for bigram in bigram_tuples:
                score += ngram_p[bigram]
        
        #trigram sentence probabilities: same as bigram, but with two start symbols
        if (n==3):
            tokens.insert(0, '*')
            tokens.insert(0, '*')
            tokens.append('STOP')
            trigram_tuples = tuple(nltk.trigrams(tokens))
            for trigram in trigram_tuples:
                score += ngram_p[trigram]
        
        #add particular sentence score to the list of scores
        scores.append(score)
         
         
    return scores


#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):
    scores = []
    
    for sentence in brown:
        
        #tokenize each sentence, add two start symbols and a STOP symbol
        tokens = nltk.word_tokenize(sentence)
        tokens.insert(0, '*')
        tokens.insert(0, '*')
        tokens.append('STOP')

        #initialize sentence score at 0
        score = 0

        #lambda of 1/3
        l = (float(1)/3)
        
        #loop through the tokens of the sentence, starting with the first token
        #that is not a start symbol, and add (to the score) the log of  the sum of each n gram probability
        #each of which is obtained by raising 2 to the log probability of that ngram
        for x in range (2, len(tokens)):
            
            tokscore=0

            if (tokens[x],) in unigrams:
                tokscore += l*(2**unigrams[(tokens[x],)])

            if (tokens[x-1], tokens[x]) in bigrams: 
                tokscore += l*(2**bigrams[(tokens[x-1], tokens[x])])

            if (tokens[x-2], tokens[x-1], tokens[x]) in trigrams:
                tokscore += l*(2**trigrams[(tokens[x-2], tokens [x-1], tokens[x])])
             
            if tokscore == 0:
                score = -1000
                break
            else:
                score += math.log(tokscore, 2)
                
             
        scores.append(score)

     
    return scores

def main():
    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

    #calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(brown)

    #question 1 output
    q1_output(unigrams, bigrams, trigrams)

    #score sentences (question 2)
    uniscores = score(unigrams, 1, brown)
    biscores = score(bigrams, 2, brown)
    triscores = score(trigrams, 3, brown)

    #question 2 output
    score_output(uniscores, 'A2.uni.txt')
    score_output(biscores, 'A2.bi.txt')
    score_output(triscores, 'A2.tri.txt')

    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')

    #open Sample1 and Sample2 (question 5)
    infile = open('Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open('Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    #score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    #question 5 output
    score_output(sample1scores, 'Sample1_scored.txt')
    score_output(sample2scores, 'Sample2_scored.txt')

if __name__ == "__main__": main()
