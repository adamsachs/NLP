import nltk
from nltk.align import IBMModel1
from nltk.align import IBMModel2

def create_ibm1(aligned_sents):
    ibm = IBMModel1(aligned_sents, 10)
    return ibm   

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm = IBMModel2(aligned_sents, 10)
    return ibm



# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    aer_sum = 0
    for i in range (0, 50):
        sentence = aligned_sents[i]
        model_alignment = model.align(sentence)
        sentence_aer = model_alignment.alignment_error_rate(sentence)
        aer_sum += sentence_aer
    
    return float(aer_sum)/50

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    f = open(file_name, 'w')
    
    for i in range (0, 20):
        sentence = aligned_sents[i]
        #source = ' '.join(sentence.words)
        source = str(sentence.words)
        f.write(source + '\n')
        #target = ' '.join(sentence.mots)
        target = str(sentence.mots)
        f.write(target + '\n')
        alignments = str(model.align(sentence).alignment)
        f.write(alignments + '\n\n')
         



def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
