import sys
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':
    sentences = sys.stdin.readlines()
    tp = TransitionParser.load(sys.argv[1])
    for sentence in sentences:
        dg = DependencyGraph.from_sentence(sentence) 
        parsed = tp.parse([dg])
        print parsed[0].to_conll(10).encode('utf-8')
        #print '\n'
