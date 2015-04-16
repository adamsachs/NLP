This is code I wrote for n-gram language modeling ('solutionsA.py') and a POS tagger ('solutions B.py') that uses the Viterbi algorithm. 



approximate time for part A: 1 min
for part B: 12 min

Part A:
Part 2 perplexity output:
A2.uni.txt:The perplexity is 1104.83292814
A2.bi.txt: The perplexity is 57.2215464238
A3.bi.txt: The perplexity is 5.89521267642

Part 3 perplexity output:
The perplexity is 13.0759217039


Part 4:
It seems that the trigram model performs better than does the linear
interpolation model, as the perplexity of the trigram model is lower than
that of the linear interpolation model, and perplexity can be thought of as
the inverse likelihood of the test sequence according to the model). This is probably due to the choice
of setting all lambda's equal. If we wanted to improve our linear
interpolation model, we would set our lambda's with a more sophisticaed
method--  based on a held-out corpus--so that they can give more weight to the models that are more accurate.


Part 5:
Output of perplexity script for Sample1.txt:
The perplexity is 11.6492786046

for Sample2.txt
The perplexity is 1611241155.03

--Thus it seems far more likely that Sample1.txt belongs to the Brown
dataset, as its lower perplexity score indicates that our model is far more
likely to produce this test sequence.


Part B:
Part 5:
Output of pos.py is:
Percent correct tags: 93.3191314715


Part6:
Output of pos.py is:
Percent correct tags: 96.9288799422

My HMM tagger performs slightly worse  than the NLTK trigram tagger with
backoff, as it tags around ~93% of the words correctly while the trigram
tagger tagged ~96% correctly
