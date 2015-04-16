This is code for a Nivre dependency parser, which parses the dependency relationships of  words in a sentence. The most interesting part of the code that I wrote is in the 'featureextractor.py' file, which extracts certain features from a corpus of sentences used in training the parser. 





1b. To determine whether a dependency graph is projective, we first make a
list of all the arcs in the dependency graph. We then go through each arc in
this list, identifying if there is any other arc that has both one end (i.e.
either a head or a dependent) that lies within the original arc and another
end that lies outside the original arc. That is, we go through each arc and
determine if there is another arc that overlaps it. If there is another arc
that overlaps it, the dependency graph is projective.


1c. Sentence with projective dependency graph: "I ate some meat."
Sentence with non-projective dependency graph: "Fred opened a door today that had no handle." 'today' is connected to 'open' while the
subsentence 'that had no handle' is connected to 'door.'



2b. 
For the swedish data set, badfeatures.model gives me these results:
UAS: 0.229038040231 
LAS: 0.125473013344

Looking at the dependency graphs produced by the badfeatures.model, almost
all arcs are unlabeled arcs coming from the artificial root at the beginning
of the sentence. For a sentence of around 10 words, there are only 2 or 3 labeled
arcs produced, which tells us that this parser does not have enough
information from the feature vector space to predict most dependency arcs of
a sentence. Additionally, it seems that of the arcs that are actually
produced, they are mainly labeled with one of a few different dependency
tags (e.g. 'ET', 'IK', 'IP'). This suggests that the features that are
employed in the badfeatures.model are more informative with regards to these
specific dependency relations, and not other relations. 

3a.
Feature #1--part of speech tag for head of stack:
The implementation of this feature is pretty straightforward. I retrieve the
'tag' field of the token that represents the head of the stack and I append
it to the results string. This feature makes a pretty noticeable difference
in the performance of the parser: to give an estimation, after removing just
this feature and running the parser on the swedish data set, performance
decreased by about 14%. This is expected--it makes sense that the
part of speech of the word on top of the stack is informative, because in each
step of the parser, we look to draw arcs between the word on top of the stack
and the word in front of the buffer. Clearly, the part of speech of one of
these words will give us information about what sort of relationship exists
between the words, and thus what move the parser should make. The time
complexity for this will just be O(n), where n is the number of words in the
sentence, as we just need to perform a single look-up for each transition,
and the amount of transitions is bounded by n (p.26 in the book). 

Feature #2--part of speech tag for first word in buffer:
The implementation of this features is also pretty straightforward. Same
as feature #1, except I retrieve the field from the token of the first word
in the buffer. The feature makes a similarly (and considerable) difference in performance  as
does feature #1: removing it reduced the performance on the swedish data set
by about 14%. This makes sense: the POS tag for the first word of the
buffer should give us similar amount of information as the POS tag for the
first word of the stack--the parser tries to determine the relationship
between these two words, so both words' POS should inform that
determination similarly. Why both (or either) of these POS tags makes a considerable
difference in performance is explained above in the feature #1
explanation. The time complexity of this feature will also be
O(n), for the same reason as described above--we'll perform a constant-time
lookup n times.


Feature #3--number of children of first word in buffer:
I figure out the number of left and right children of the first word of the
buffer by cycling through the arcs that have already been drawn, and only
selecting those whose first value in the tuple (i.e. the head) is the same
as the index of the first word in the buffer. I then take those, and
partition them into those whose child's index is less than the index of the first
word in the buffer, and those whose child's index is more--which corresponds
to left and right children respectively. This gives me counts for the right
and left children, and then I append those values to the result string. This
feature does not have nearly as big of an effect on performance as does the
POS tags. When removed from the feature extractor, it reduced the
performance of the parser by around 2%. It is not hard to see that simply a
count of the number of left/right children would not have as big of an
effect on performance as does the POS tags of the first word in the
buffer/top of the stack. POS tags of these two words  have a more direct correlation with the
next move of the parser (as it determines whether/which relation to draw
btwn them) than does the right and left children of the first word in the
buffer--which gives some information about the nature of the first word in
the buffer (i.e how many words to the left and right depend on it), but not
nearly as much as the POS tag. This is reflected in the slight increase in
performance--the left/right children does tell us something (but not
that much) about the first word in the buffer. The time complexity for this
feature is upward-bound by O(n^2), where n is the amount of words in the
sentence, because for every n transition, there are a possible n amount of
children that need to be counted. 


performance (i.e. LAS score) of all features:

on swedish: 0.686
on danish: 0.719
on korean: 0.617
on english: 0.734

3d.
The deterministic, arc-eager parser with projective sentences has a time
complexity of O(n), where n is the number of words in the sentence (pg. 26
of the book). This is because the amount of transitions is bounded by n, as
no transition increases the length of the overall amount of words we are
dealing with (i.e. the sum of the length of the stack and buffer). And we
assume here that each transition is computed in constant time (which might
not be the case for the oracle, if the feature computations are overly
complex). 

The model here limits us to projective sentences, though--and not all sentences we are dealing with are
necessarily projective. This requires pre/post processing of these sentences
(or simply just ignoring them, as we do here).



