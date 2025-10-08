'''
    Generator for the numerator and denominator lattices of the first value of the rational solutions for cycles satisfying the Collatz criteria.

    We work from integer matrices of rank N that provide rational solutions of cycle-length N-1.

'''

import itertools
import math
import numpy as np
from sympy.functions.special import tensor_functions

'''
    State vector for lattice is:
    [n, a, b, t]
    n : current numerator
    a + b : current denominator
    t : addend for next numerator 
'''
nd_label_root = "_"
nd_vec_root = np.array([0, -1, 1, -1])

nd_op_up = np.array([
    [-2,   0,  0,  1 ],
    [ 0,  -3,  0,  0 ],
    [ 0,   0, -2,  0 ],
    [ 0,   0,  0, -3 ]
])

nd_op_dn = np.array([
    [-1,  0,  0,  0 ],
    [ 0, -1,  0,  0 ],
    [ 0,  0, -2,  0 ],
    [ 0,  0,  1, -1 ]
])

nd_op_invdn = np.array([
    [-1,      0,      0,  0 ],
    [ 0,     -1,      0,  0 ],
    [ 0,      0, -1.0/2,  0 ],
    [ 0,      0, -1.0/2, -1 ]
])

def closest_int(f):
    if 0.0 <= f:
        return int(f + 0.49999999999999999999999999999999999)
    else:
        return int(f - 0.50000000000000000000000000000000000)

vf_closest_int = np.vectorize(closest_int)


def rotateLabelBits(label_bits):
    ''' Rotate  a string e.g. "001" -> "010"
    '''
    car = label_bits[0]
    cdr = label_bits[1:]
    return cdr + car
#

def getRotatedLabelSet(label_bits):
    ''' 
        Given one rotatable string, collect the whole set.  
        (We use set() to filter out the symmetric cases if any)
    '''
    n = len(label_bits)
    rotations = set([label_bits])
    for i in range(n):
        label_bits = rotateLabelBits(label_bits)
        rotations.add(label_bits)
        
    return rotations
#

def labelFromInt(generation, val):
    return "_" + format(val, 'b').zfill(generation)
#


def getSolutionSet(generation):
    def rotateTillDup(numbits, to_rotate):
        collect = set()
        while to_rotate not in collect:
            collect.add(to_rotate)
            rac = to_rotate & 1
            if rac == 1:
                to_rotate =  (to_rotate >> 1) | (2**(numbits-1))
            else:
                to_rotate =  (to_rotate >> 1) 
        return collect 
    def i2l(val):
        return labelFromInt(generation, val)
    sets = []
    vals = set(range(2**generation))
    while len(vals) > 0:
        s = rotateTillDup(generation, vals.pop())
        sets.append(s)
        vals = vals - s
    result = []
    for ss in sets:
        result.append(list(map(i2l, ss)))

    return result
#

def matrixFromLabel(label):
    rank = len(label)
    mat = np.zeros((rank,rank))
    y = np.zeros((rank))
    mat[rank-1][0] = 1
    mat[rank-1][rank-1] = -1
    for i in range(rank-1):
        mat[i][i+1] = 2
        if label[i+1:i+2] == "1":
            mat[i][i] = -3
            y[i] = 1
        else:
            mat[i][i] = -1
    return mat, y

def solutionFromLabel(label):
    mat, y = matrixFromLabel(label)
    x = np.linalg.solve(mat, y)
    det = np.linalg.det(mat)
    numer = closest_int(x[-1]*det)
    denom = closest_int(det)
    return (numer, denom), x

def numerMatrixFromLabel(label, column):
    mat, y = matrixFromLabel(label)
    mat.T[column] = y
    return mat

#
def matrixIndexVals(M, indexes):
    rank = M.shape[1]
    v = np.zeros(rank)
    row = 0
    for idx in indexes:
        v[row] = M[row, idx]
        row = row +1
    return v
#
def determinantParts(M):
    parts = []
    rank = M.shape[1]
    for p in itertools.permutations(list(range(rank))):
        sign = tensor_functions.eval_levicivita(*p)
        t = (sign, *matrixIndexVals(M, p))
        if abs(math.prod(t))> 0.000001:
            parts.append(t)
    return parts
#

class Node:
    def __init__(self, label, vec):
        ''' label: a string like "_101" that identifies the node position in the binary tree
            vec:  a numpy array of 4 values: [n, a, b, t]
                   n: the current numerator
                   a: the denominator "3-factor"
                   b: the denominator "2-factor"
                   t: the carried addend for future numerator calculations
        '''
        self.label = label
        self.vec = vec 
        self.rank = len(label)
    
    def down_vec(self):
        ''' Here we see that:
                the Up numerator of rank+1 is a function of the "2-factor" of the parent of parent denominator of rank-1
                t: the t computed here becomes a factor in the _next_ Up numerator
                b: the b that gets added to t is the "2-factor" of the _previous_ denominator
        '''
        forward_down = np.inner(nd_op_dn, self.vec)
        #n,a,b,t = self.vec
        #a_, b_ = (-1)*a, (-2)*b
        #n_ = (-1)*n
        ## p = self.rank - 1 ... 2**(p) :=: b (on entry)
        #t_ = (-1)*t + b                                 # <-- Note this relationship between numerator and denominator
        #forward_down = np.array([n_, a_, b_, t_])
        return forward_down

    def up_vec(self):
        forward_up = np.inner(nd_op_up, self.vec)
        #n,a,b,t = self.vec
        #a_, b_ = (-3)*a, (-2)*b
        #n_ = (-2)*n + t
        #t_ = (-3)*t
        #forward_up = np.array([n_, a_, b_, t_])
        return forward_up

    def invdn_vec(self):
        rev_dn = np.inner(nd_op_invdn, self.vec)
        #n_,a_,b_,t_ = self.vec
        #a, b = (-1)*a_, b_//(-2)  # b is always a multiple of 2
        #c = b
        #t = (-1)*(t_ - c)
        #n = (-1)*n_
        #rev_dn = np.array([n, a, b, t])
        return rev_dn

    def invup_vec(self):
        # we can't use a simple single step transform ... 
        n_, a_, b_, t_ = self.vec
        a, b = a_//(-3), b_//(-2)
        t = t_//(-3)
        n = (n_ - t)//(-2)  # subtraction followed by division can't be done in one matrix step
        return np.array([n, a, b, t])

    def forward(self, zero_for_down_one_for_up):
        if zero_for_down_one_for_up == 0:
            return Node(self.label + "0", self.down_vec())
        else:
            return Node(self.label + "1", self.up_vec())

    def reverse(self, zero_for_down_one_for_up):
        if zero_for_down_one_for_up == 0:
            return Node(self.label[0:-1], self.invdn_vec())
        else:
            return Node( self.label[0:-1],self.invup_vec())
        
    def numerator(self):
        return self.vec[0]

    def denominator(self):
        return self.vec[1] + self.vec[2]

    def tup(self):
        return (self.numerator(), self.denominator())

    def solution_set_size(self):
        return len(getRotatedLabelSet(self.label[1:]))

    def solveMatrix(self):
        ''' Solve the rational equation for the current node's cycle values.  
            x[0] of the solution should always be numerator/denominator since the lattice is a generator for the x[0]
        '''
        return solutionFromLabel(self.label)
    #

    def t_sym(self):
        ''' Generate t as a symbolic string
            One obvious intersting charateristic:
              dn() is essentially a subtraction operation
              up() is essentially a multiplication operation
        '''
        if self.label == "_":
            return "(-1)"
        elif self.label[-1] == "0":
            parent = self.reverse(0)
            return "(%d - %s)"%(parent.vec[2], parent.t_sym())
        else:
            parent = self.reverse(1)
            return "(-3 * %s)"%(parent.t_sym())

    def t_prev_sym(self):
        ''' Get the prior t value that fed into this node as a symbolic string
        '''
        if self.label == "_":
            return "!"
        elif self.label[-1] == "0":
            return "0"
        else:
            return self.reverse(1).t_sym()
    #        

    def numerMatrix(self, column):
        return numerMatrixFromLabel(self.label, column)

    def Dx0_Parts(self):
        ''' Extract the x_0 determinant parts which comprise the x_0 numerator as a list of tuples such that:
            D_x_0 = SUM(PROD(tup)) for tup in tuples.

            Reason: Trying to eyeball relationship of parts of n, b, t
        '''
        return determinantParts(self.numerMatrix(0))
#
class NodeRoot(Node):
    ''' The starting point for the lattices
    '''
    def __init__(self):
        super().__init__(nd_label_root, nd_vec_root)
#

def nodeFromLabel(label):
    n = NodeRoot()
    for digit_str in label[1:]:
        up_dn = int(digit_str)
        n = n.forward(up_dn)
    return n
#


