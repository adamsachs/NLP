class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        #raise NotImplementedError('Please implement left_arc!')
        
        #creates list of all dependents that have arcs going to them
        jlist = []
        for arc in conf.arcs:
            jlist.append(arc[2])
        
        #preconditions are that top of stack isn't root 0 and doesnt have a head
        #i.e. it is not a dependent of an arc
        if conf.stack[-1] is 0 or conf.stack[-1] in jlist:
            return -1

        idx_wi = conf.buffer[0]
        idx_wj = conf.stack.pop()

        conf.arcs.append((idx_wi, relation, idx_wj))

        
    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))
        
        
    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        #raise NotImplementedError('Please implement reduce!')
        
        #creates list of all dependentes that have arcs going to them
        jlist = []
        for arc in conf.arcs:
            jlist.append(arc[2])
        
        #precondition is that top of stack has a head already
        if not conf.stack[-1] in jlist:
            return -1

        conf.stack.pop()
        
        
    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        #raise NotImplementedError('Please implement shift!')
        
        #only precondition is that there is something in the buffer
        if not conf.buffer:
            return -1
        

        conf.stack.append(conf.buffer.pop(0))


