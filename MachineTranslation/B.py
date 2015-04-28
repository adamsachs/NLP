import nltk
import A
from nltk.align import AlignedSent
from collections import defaultdict

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        
        alignment = []
        
        germ_sent = [None] + align_sent.words
        en_sent = [None] + align_sent.mots

        l = len(germ_sent)
        m = len(en_sent)

        i = 0
        for word in germ_sent:

            #initialize max_prob to null alignment (i.e. target = None, j = 0)
            max_prob = ((self.t['german'][word][None][0]*self.q['german'][(i, l, m)][0][0]) +
                             self.t['english'][None][word][0]*self.q['english'][(0, m, l)][i][0]) / float(2) 

            winner = None
            
            j=0
            for target in en_sent:
                curr_prob = ((self.t['german'][word][target][0]*self.q['german'][(i, l, m)][j][0]) +
                             self.t['english'][target][word][0]*self.q['english'][(j, m, l)][i][0]) / float(2)
                

                if curr_prob > max_prob:
                    max_prob = curr_prob
                    winner = j
                
                j +=1 
            
            #add alignment if it is not aligned to NULL
            if  winner != None and i != 0:
                alignment.append((i-1, winner-1))
            
            i += 1
        
        
        return AlignedSent(align_sent.words, align_sent.mots, alignment)

    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        
        t = {}
        q = {}
        
        t['german']= {}
        t['english'] = {}
        q['german'] = {}
        q['english'] = {}
        


        #initialize parameters uniformly
        for sentence in aligned_sents:
            
            en_sent = [None] + sentence.mots
            germ_sent = [None] + sentence.words

            i = 0
            l = len(en_sent)
            m = len(germ_sent)
            
            
            for word in en_sent:
                if not t['english'].has_key(word):
                    t['english'][word] = {}
                if not q['english'].has_key((i, l, m)):
                    q['english'][(i, l, m)] = {}
                
                j = 0
                for target in germ_sent:
                    if not t['english'][word].has_key(target):
                        
                        t['english'][word][target] = []
                        
                        #will be t(target|word)
                        t['english'][word][target].append(0)
                    
                    #initialize q values to 1/l
                    if not q['english'][(i, l, m)].has_key(j):
                        q['english'][(i, l, m)][j] = []
                        q['english'][(i, l, m)][j].append(float(1)/l)


                    j += 1

                i += 1
            




            i = 0
            l = len(germ_sent)
            m = len(en_sent)
            for word in germ_sent:
                
                if not t['german'].has_key(word):
                     t['german'][word] = {}
                if not q['german'].has_key((i, l, m)):
                    q['german'][(i, l, m)] = {}

                j = 0    
                for target in en_sent:
                    if not t['german'][word].has_key(target):
                        t['german'][word][target] = []

                        #will be t(target|word)
                        t['german'][word][target].append(0)
                   

                    #initalize q values to 1/l
                    if not q['german'][(i, l, m)].has_key(j):
                        q['german'][(i, l, m)][j] = []
                        q['german'][(i, l, m)][j].append(float(1)/l)
                   
                    j += 1
                
                i += 1


            
            #initialize t values uniformly
            for language in t:
                for word in t[language]:
                    count = len(t[language][word])
                    for target in t[language][word]:
                        t[language][word][target][0] = float(1)/count
                         
                   

#iterate num_iter times:
        for x in range(0, num_iters): 
            
            for sentence in aligned_sents:

                en_sent = [None] + sentence.mots
                germ_sent = [None] + sentence.words
                

                                
                #english to german counts
                i = 0
                l = len(en_sent)
                m = len(germ_sent)
                for word in en_sent:
                    
                    #initialize the word's count if first treatment of word
                    if not t['english'][word].has_key('word_sum'):
                        t['english'][word]['word_sum'] = 0
                    
                    #initialize (i, l, m) count if first treatment of indices/parameters
                    if not q['english'][(i, l, m)].has_key('ilm_sum'):
                        q['english'][(i, l, m)]['ilm_sum'] = 0

                    #used for normalization over sentence
                    wordtargsum = 0
                    targwordsum = 0

                    j = 0
                    for target in germ_sent:
                        reverse_params = (j, m, l)
                        wordtargsum += t['english'][word][target][0]*q['english'][(i, l, m)][j][0]
                        targwordsum += t['german'][target][word][0]*q['german'][reverse_params][i][0]
                        
                        #initalize word/target t count to 0 if first treatment of word:target pair
                        if len(t['english'][word][target]) < 2:
                            t['english'][word][target].append(0)
                        
                        #initalize (j | i, l, m)  count to 0 if first reatment
                        if len(q['english'][(i, l, m)][j]) < 2:
                            q['english'][(i, l, m)][j].append(0) 
                        
                        j += 1
                    
                    j = 0
                    for target in germ_sent:
                        reverse_params = (j, m, l)
                        non_normalized1 = (t['english'][word][target][0]*q['english'][(i, l, m)][j][0])
                        non_normalized2 = t['german'][target][word][0]*q['german'][reverse_params][i][0]
                        
                        #normalization over the sentece
                        delta1 = non_normalized1/float(wordtargsum)
                        delta2 = non_normalized2/float(targwordsum)
                        delta = (delta1 + delta2)/2.0
                        
                        t['english'][word][target][1] +=  delta 
                        t['english'][word]['word_sum'] +=  delta 
                        q['english'][(i, l, m)][j][1] += delta
                        q['english'][(i, l, m)]['ilm_sum'] += delta
                        
                        j += 1

                    i += 1

                #german to english counts
                i = 0
                l = len(germ_sent)
                m = len(en_sent)
                for word in germ_sent:

                    #initialize the word's count if first treatment of word
                    if not t['german'][word].has_key('word_sum'):
                        t['german'][word]['word_sum'] = 0
                    
                    #initialize (i, l, m) count if first treatment of indices/parameters
                    if not q['german'][(i, l, m)].has_key('ilm_sum'):
                        q['german'][(i, l, m)]['ilm_sum'] = 0
                        

                    #used for normalization over sentence
                    wordtargsum = 0
                    targwordsum = 0

                    j=0
                    for target in en_sent:
                        reverse_params = (j, m, l)
                        wordtargsum += t['german'][word][target][0]*q['german'][(i, l, m)][j][0]
                        targwordsum += t['english'][target][word][0]*q['english'][reverse_params][i][0]
                        
                        #initalize word/target count to 0 if first treatment of word:target pair
                        if len(t['german'][word][target]) < 2:
                            t['german'][word][target].append(0)
                        
                        #initialize (j| i, l, m) count to 0 if first treatment
                        if len(q['german'][(i, l, m)][j]) < 2:
                            q['german'][(i, l, m)][j].append(0)
                        
                        j += 1
                    
                    j = 0
                    for target in en_sent:
                        reverse_params = (j, m, l)
                        non_normalized1 = t['german'][word][target][0]*q['german'][(i, l, m)][j][0]
                        non_normalized2 = t['english'][target][word][0]*q['english'][reverse_params][i][0]
                        
                        #normalization over the sentece
                        delta1 = non_normalized1/float(wordtargsum)
                        delta2 = non_normalized2/float(targwordsum)
                        delta = (delta1 + delta2) / 2.0

                        t['german'][word][target][1] += delta 
                        t['german'][word]['word_sum'] += delta
                        q['german'][(i, l, m)][j][1] += delta
                        q['german'][(i, l, m)]['ilm_sum'] += delta
                        
                        j += 1
      
                    
                    i += 1
            
            #recalculate parameters--normalization over the word
            
            #english
            for word in t['english']:
                for target in t['english'][word]:
                    if not target == 'word_sum':
                        
                        t['english'][word][target][0] = t['english'][word][target][1]/float(t['english'][word]['word_sum'])
                        
                        

            for ilm in q['english']: 
                for j in q['english'][ilm]:
                    if not j == 'ilm_sum':
                        q['english'][ilm][j][0] = q['english'][ilm][j][1]/float(q['english'][ilm]['ilm_sum']) 
           
            #german
            for word in t['german']:
                for target in t['german'][word]:
                    if not target == 'word_sum':
                        t['german'][word][target][0] = t['german'][word][target][1]/float(t['german'][word]['word_sum'])

                       

            for ilm in q['german']: 
                for j in q['german'][ilm]:
                    if not j == 'ilm_sum':
                        q['german'][ilm][j][0] = q['german'][ilm][j][1]/float(q['german'][ilm]['ilm_sum'])
                    
                        
           


                
        return (t,q)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
