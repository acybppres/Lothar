import sys, io
import math
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from fractions import Fraction
from sympy import factorint
from itertools import product, combinations
from typing import List, Optional, Tuple
import random

def C(n):
    """ Compute next value in simplified Collatz sequence.
    """
    if n & 1 == 0:
        return n//2
    else:
        return (3*n + 1)//2
#
def L_C(n):
    """ Compute binary label-string for a given Collatz number
    """
    if n == 1:
        return "1"

    S = ""
    while n != 1:
        if n & 1 == 0:
            n = n//2
            S = S + "1"
        else:
            n = (3*n + 1)//2
            S = S + "0"
    return S
#
def collatzVector(collatzNumber):
    chain = [collatzNumber]
    while collatzNumber != 1:
        if (collatzNumber & 1) == 0:
            collatzNumber = collatzNumber // 2
        else:
            collatzNumber = (3 * collatzNumber + 1) // 2
        chain.append(collatzNumber)
    chain.append(chain[-2])
    return np.array(chain)
    
def collatzPath(collatzNumber):
    path = []
    while collatzNumber != 1:
        if (collatzNumber & 1) == 0:
            collatzNumber = collatzNumber // 2
            path.append("1")
        else:
            collatzNumber = (3 * collatzNumber + 1) // 2
            path.append("0")
    return "".join(path)
#
def Ay_L(L):
    """ Generate A matrix and y vector from label string
    """
    rank = len(L) + 2    
    A = np.zeros((rank,rank))
    y = np.zeros((rank))
    for row in range(rank-3):
        if L[row] == "1":
            a_val = -1.0
            y_val = 0.0
        else:
            a_val = -3.0
            y_val = 1.0
        A[row][row] = a_val
        A[row][row+1] = 2.0
        y[row] = y_val
    #
    # Last 3 rows are always the same
    row = rank - 3
    A[row][row] = -1
    A[row][row+1] = 2
    y[row] = 0
    row = rank - 2
    A[row][row] = -3
    A[row][row+1] = 2
    y[row] = 1
    row = rank - 1
    A[row][row] = -1
    A[row][row-2] = 1
    y[row] = 0
    
    return A, y
#
def solve_Ay_L(L):
    """ Solve for the x vector given the label-string
    """
    A, y = Ay_L(L)
    return A, np.linalg.solve(A, y), y
#
def x0_L(L):
    """ Get the x[0] value given a label-string
    """
    A, x, y = solve_Ay_L(L)
    return round(x[0])  # clean up mantisa garbage
#
def countZeros(label):
    zero_count = 0
    for bit in label:
        if bit == "0":
            zero_count += 1
    return zero_count
#

def Z(L):
    """ Indexes of zeros in label string
    """
    for i in range(len(L)):
        if L[i] == "0":
            yield i
#
def a_b_c_S(L):
    """ Get the (power-of-two, power-of-three, zero-sum-accumulator) tuple for a node given its label
    """
    a = len(L)
    b = 0
    for bit in L:
        if bit == "0":
            b += 1
    ZZ = [(j,i) for j, i in enumerate(Z(L))]
    c = sum((3 ** (b - j - 1)) * (2 ** (i)) for j, i in ZZ)
    S = [zz[1] for zz in ZZ]
    return (a,b,c,S)
#
def val_a_b_c(a_b_c):
    """ Get the value for a node given the tuple (power-of-two, power-of-three, zero-sum-accumulator)
    """
    a, b, c = a_b_c
    f = Fraction( ((2**a) - c), (3**b) )
    return (f.numerator, f.denominator)
#
def val_a_b_c_S(a_b_c_S):
    a_b_c = (a_b_c_S[0], a_b_c_S[1], a_b_c_S[2])
    return val_a_b_c(a_b_c)
#
def val_L(L):
    """ Get the value for a node given the label string
    """
    return val_a_b_c_S(a_b_c_S(L))
#
N_i = ((0,0), [])
def mr_TupItemValue(a_b, a_0):
    """ Value of (a, b) tuple with a_0 added to b so we are not dealing with float limitations
    """
    a,b = a_b
    return (2**a)*(3**(a_0 + b))
#
def mrTupValue(mr_tup):
    """ Compute the value of the given mrTup

        Returns value as a cannonical (numerator, denominator) pair
    """
    # Multiplying the numerator by 3 ** the generation keeps us in integer land
    a_0 = mr_tup[0][0]
    total = mr_TupItemValue(mr_tup[0], a_0)
    for a_b in mr_tup[1]:
        total -= mr_TupItemValue(a_b, a_0)
    frac = Fraction(total, 3**a_0)
    return (frac.numerator, frac.denominator)
#
# Terse version for math conversion
def F_0(mr_tup):
    return ( (mr_tup[0][0]+1, mr_tup[0][1]-1), mr_tup[1] + [(mr_tup[0][0], mr_tup[0][1]-1)] )
#
def F_1(mr_tup):
    return ((mr_tup[0][0]+1, mr_tup[0][1]), mr_tup[1])
#
def F_rev(mr_tup):
    """ Reverse of either F0 or F1, uses contents of L list to determine COA
    """
    u_tup, v_list = mr_tup
    a,b = u_tup
    a_ = a-1
    b_ = b
    if len(v_list) > 0:
        if v_list[-1][0] == a_:
            b_ = b+1
            if len(v_list) > 1:
                v_list =  [(vt[0], vt[1]) for vt in v_list[0:-1]]
            else:
                v_list = []
    return ((a_, b_), v_list)
#
"""
G_0, G_1 work from the other side of the label prepending to the label

These parallel the Collatz Sequence
"""
def G_0(mr_tup):
    return ( (mr_tup[0][0]+1, mr_tup[0][1]-1), [(0,-1)] + [(c+1, d-1) for (c, d) in mr_tup[1]] )
#
def G_1(mr_tup):
    return ((mr_tup[0][0]+1, mr_tup[0][1]), [(c+1, d) for (c, d) in mr_tup[1]])
#
def G_rev(mr_tup):
    """ Reverse of either G_0 or G_1, uses contents of L list to determine COA
    """
    u_tup, v_list = mr_tup
    a,b = u_tup
    a_ = a-1
    b_ = b
    if len(v_list) > 0:
        if v_list[0][0] == 0 and v_list[0][1] == -1:
            # Then we are undoing a G_0 ...
            b_ = b+1
            if len(v_list) == 1:
                v_list = []
            else:
                v_list = [(c-1, d+1) for (c, d) in mr_tup[1][1:]]
        else:
            v_list = [(c-1, d) for (c, d) in mr_tup[1]]
    return ((a_, b_), v_list)
#
def mrTupFromPath(label):
    """ Create an mrTup given a path (label)
    """
    mr_tup = N_i
    for bit in label:
        if bit == "1":
            mr_tup = F_1(mr_tup)
        else:
            mr_tup = F_0(mr_tup)
    return mr_tup
#
def mrTupFromValue(n):
    label = collatzPath(n)
    return mrTupFromPath(label)
#
def mrTupToLaTex(T):
    a, b = T[0]
    s = "\\frac{2^{%d}}{3^{%d}}"%(a, -b)
    L = T[1]
    if len(L) > 0:
        s = s + " - ( "
        plus = "  "
        for c_d in L:
            c, d = c_d
            t = "\\frac{2^{%d}}{3^{%d}}"%(c, -d)
            s = s + plus + t
            plus = " + "
        s = s + " )"
    return "$ " + s + " $"
#
def rationalCollatzPath(fraction_tup, max_length=100):
    """ Apply Collatz rules to a rational to see if it has a path to 1 and is in the lattice 
    """
    numerator_0, denominator_0 = fraction_tup
    rational_collatz_path = [(numerator_0, denominator_0)]
    numerator, denominator = fraction_tup
    for i in range(max_length):
        if numerator == 1 and denominator == 1:
            break
        if (abs(numerator) & 1) == 1:
            numerator = 3*numerator + denominator
        denominator = 2*denominator
        f = Fraction(numerator, denominator)
        numerator = f.numerator
        denominator = f.denominator
        # Check for a cycle
        if (numerator, denominator) in rational_collatz_path:
            rational_collatz_path.append((numerator, denominator))
            break
        rational_collatz_path.append((numerator, denominator))

    return rational_collatz_path
#
def gen_generation(a):
    """ Generate a (label, a_b_c_tuple, numerator_denominator_pair) tuple for each
        lattice node in a generation

        Note: uses what we have learned regarding the mapping of the label string
        to create a nodes L subtractand list -- does not require applying F_0, F_1

        Fastest way to generate a generation's nodes
    """
    seqs = product('10', repeat=a)
    for bits in seqs:
        label = ''.join(bits)
        zeros = [i for i, b in enumerate(bits) if b == '0']
        b = len(zeros)
        # compute c = sum_{j=0}^{k} 3^{k-j} * 2^{i_j - 1}
        c = sum((3 ** (b - j - 1)) * (2 ** (i)) for j, i in enumerate(zeros))
        f = Fraction(2**a - c, 3**b)
        yield (label, (a,b,c), (f.numerator, f.denominator))
#
def mr2Nplus_1(T):
    B = len(T[1])  
    L = [(0, -1)]

    # Keep initial zeros
    idx = 0
    for idx, val in enumerate(T[1]):
        if T[1][idx][0] != idx:
            break
        L.append( (T[1][idx][0] + 1, T[1][idx][1]-1) )
    # Remove the first tuple where (a, -a) is true
    match = False
    for i in range(idx, B, 1):
        if (not match) and (T[1][i][0] == -T[1][i][1]):
            match = True
        else:
            L.append( (T[1][i][0]+1, T[1][i][1]) )
    if not match:
        return None
    return ( (T[0][0] + 1, T[0][1]), L)
#
def mrTupToLaTex(T):
    a, b = T[0]
    s = "\\frac{2^{%d}}{3^{%d}}"%(a, -b)
    L = T[1]
    if len(L) > 0:
        s = s + " - ( "
        plus = "  "
        for c_d in L:
            c, d = c_d
            t = "\\frac{2^{%d}}{3^{%d}}"%(c, -d)
            s = s + plus + t
            plus = " + "
        s = s + " )"
    return "$ " + s + " $"
#

def generationLabels(a):
    """
    Yields the 2^a label strings of the labels/prefixes for a given generation
    """
    if a == 0:
        return ""
    seqs = product('10', repeat=(a))
    for bit_tup in seqs:
        label = "".join(bit_tup)
        yield label
#


def shortParity(k, n):
    parity_bits = []
    for i in range(k):
        if n == 1:  
            return None
        n_ = C(n)
        if n_ < n:
            parity_bits.append("0")
        else:
            parity_bits.append("1")
        n = n_
    return "".join(parity_bits)
# 
def verifyShortParityMod(k, test_count=2**10):
    mod_base = 2**k
            
    # generate 2^a slots for modulus mappings
    mapping = [""]*(mod_base)
    for i in range(mod_base, test_count+mod_base): # made sure starting label isn't shorter than test target length
        mod = i % mod_base
        parity_str = shortParity(k, i)
        if parity_str is not None:  # A number was given that has a shorter path than k
            if len(mapping[mod]) > 0:
                if parity_str != mapping[mod]:
                    print("Non-unique label map!")
            else:
                # First time
                mapping[mod] = parity_str
#
def verifyUniqueMapping(prefix_length, test_count=2**10):
    mod_base = 2**prefix_length
            
    # generate 2^a slots for modulus mappings
    mapping = [""]*(mod_base)
    for i in range(mod_base, test_count+mod_base): # made sure starting label isn't shorter than test target length
        mod = i % mod_base
        label = collatzPath(i)
        if len(mapping[mod]) > 0:
            if label[0:prefix_length] != mapping[mod]:
                print("Non-unique label map!")
        else:
            # First time
            mapping[mod] = label[0:prefix_length]
#
def T_010_to_T_111(k):  # 1(mod 8)
    return 9*k + 7
#
def T_010_to_T_110(k):
    return 3*k + 1
#
def T_010_to_T_011(k):
    return 3*k + 2
#
def T_101_to_T_111(k):
    return 3*k + 2
#
def T_001_to_T_111(k): # 3(mod 8)
    return 9*k + 5
#
def T_001_to_T_101(k):
    return 3*k + 1
#
def T_110_to_T_111(k):
    return 3*k + 4
#
def T_110_to_T_011(k):
    return k + 1
#
def T_011_to_T_111(k): # 5(mod 8)
    return 3*k + 1
#
def T_011_to_T_110(k):
    return k - 1
#
def T_011_to_T_100(k):
    # Not affine at 3 bits ...
    if (k%24) == 21:
        return (k//3) -1
    else:
        return k+2
#
def T_100_to_T_111(k):
    return 9*k + 10
#
def T_100_to_T_110(k):
    return 3*k + 2
#
def T_100_to_T_011(k):
    return 3*k + 3
#
def T_000_to_T_111(k):  # 7 (mod 8)
    return 27*k + 19
#
def T_000_to_T_110(k):
    return 9*k + 5
#
def T_000_to_T_011(k):
    return 9*k + 6
#
def T_000_to_T_100(k):
    return 3*k + 1
#

def checkMod8Formula(mod_8, tranform_func, prefix):
    for i in range(1000):
        k = 8*(i+2) + mod_8
        label = collatzPath(k)
        label_ = prefix + label[len(prefix):]
        val_ = mrTupValue(mrTupFromPath(label_))
        if val_[1] != 1:
            print(f"Label substitution FAILED {k}({label}) -> ({label_}) has noninteger {val_}")
            return False
        k_ = tranform_func(k) 
        if k_ != val_[0]:
            print(f"{tranform_func.__name__} FAILED {k} should -> {val_} but function gave {k_}")
            return False
    return True
#
def computeNextPrefixBit(a, p2, label, mod):
    # We cannot choose two label bits, so we generate an exemplar
    example = p2 + mod 
    label = collatzPath(example) 
    return a+1, 2**(a+1), label[0:a]
    
def computePrefix(n):
    mod = n % 4
    label = ["11", "01", "10", "00"][mod]
    if n > 3:
        a = 3
        p2 = 2**(a)
        while  p2 < n:
            a, p2, label = computeNextPrefixBit(a, p2, label, n % p2)
    return label
#
def mapByPrefix(prefix, x):
    """
    Given the prefix of a number (computed from its (mod 2**len(prefix))
    Compute its mapped value in the 111* partition of the lattice
    """
    a = len(prefix)
    b = countZeros(prefix)
    accum = 0
    sgn = -1**a
    for i in range(a-1):
        sgn = sgn * -1
        P = [2]*(a) # we do not do the zz term
        if prefix[i] == "0":
            P[i] = 1
        else:
            continue  # y = 0, so product will be 0
        for j in range(i+1, a, 1):
            if prefix[j] == "0":
                P[j] = -3
            else:
                P[j] = -1
        #print((sgn, P))
        accum += (sgn * np.prod(P))

    if prefix[-1] == "0":
        #print((-sgn, [2]*(a-1)))
        accum -= (sgn * (2**(a-1)))
    #print(b, accum)
    return int((3**b) * x + accum) # get rid of np.int before return
#
def mapNumberUp(n):
    """ return the 0 (mod 2^a) number that is the transform of a number from a different part of the lattice
    """
    prefix = computePrefix(n)
    n_ = mapByPrefix(prefix, n)
    return n_
#
def applyMod8Affine(n):
    def NOOP(n):
        return n
    #
    LU = {
        0: NOOP, # prefix "111",
        1: T_010_to_T_111, # prefix "010",
        2: T_101_to_T_111, # prefix "101"
        3: T_001_to_T_111, # prefix "001"
        4: T_110_to_T_111, # prefix "110"
        5: T_011_to_T_111, # prefix "011"
        6: T_100_to_T_111, # prefix "100"
        7: T_000_to_T_111  # prefix "000"
    }
    mod_8 = n % 8
    return (LU[mod_8])(n)
#
def divide2n(n):
    while (n & 1) == 0:
        n >>= 1
    return n
#
def computePrefixForModClass(power_of_2, n):
    mod = n % 4
    label = ["11", "01", "10", "00"][mod]
    if power_of_2 > 2:
        a = 3
        p2 = 2**(a)
        while a <= power_of_2:
            a, p2, label = computeNextPrefixBit(a, p2, label, n % p2)
    return label
#

def affineFunctionParamsFromPrefix(prefix):
    """
    Return the affine parameters A, B for n' = A*n + B

    (Logic extracted from earlier mapByPrefix function)
    """
    a = len(prefix)
    b = countZeros(prefix)
    accum = 0
    sgn = -1**(a)
    for i in range(a-1):
        sgn = sgn * -1
        P = [2]*(a) # we do not do the zz term
        if prefix[i] == "0":
            P[i] = 1
        else:
            continue  # y = 0, so product will be 0
        for j in range(i+1, a, 1):
            if prefix[j] == "0":
                P[j] = -3
            else:
                P[j] = -1
        #print((sgn, P))
        accum += (sgn * np.prod(P))

    if prefix[-1] == "0":
        #print((-sgn, [2]*(a-1)))
        accum -= (sgn * (2**(a-1)))
    #print(b, accum)
    return(3**b, int(accum)) # get rid of np.int before returning
#
def affineFunctionFromModulus(power_of_2, mod):
    """
    Generates the correct An+B function for the given modulus and base.
    """
    mod_base = 2**power_of_2
    prefix = computePrefixForModClass(power_of_2, mod+mod_base) # Avoid zero, one conditions with added mod_base
    A, B = affineFunctionParamsFromPrefix(prefix)
    # print(f"{mod}(mod {mod_base}): A={A}, B={B}")
    def an_affine_func(n):
        return A*n + B 
    #
    return an_affine_func
#
def maxOddPowerOf2(n):
    p2 = math.floor(math.log(n, 2.0))
    if p2 & 1 != 1:
        p2 -=1
    if p2 < 3:
        p2 = 3
    return p2
#
def maxConverge(n):
    trace = []  # Collect A, B tup and power of 2 reduced for each step
    while n != 1:
        p2 = maxOddPowerOf2(n)
        prefix = computePrefixForModClass(p2, n)
        A, B = affineFunctionParamsFromPrefix(prefix)
        n = A*n + B
        j = 0
        while n & 1 == 0:
            n >>=1
            j+=1
        trace.append(( len(collatzPath(n)), p2, (A, B), j, n, n%3))
    return trace
#

def countTwos(n):
    i = 0
    while (n & 1) == 0:
        n >>= 1
        i+=1
    return i
#
def distributionAverage(D):
    accum = 0
    count = 0
    R = {0: 0.0, 2: 0.0, 4: 0.0, 6: 0.0}
    for mod in D:
        for key in D[mod]:
            count += D[mod][key]
            accum += (key * D[mod][key])
        avg = accum / count
        R[mod] = avg
    return R
#
def distributionOfTwos(a):
    """
    For the given generation, count how many twos can divided out of a given number:
    """
    D = {}
    just_3 = []
    def collectCount(val, count):
        if count == 3:
            reduce = val
            while (reduce & 1 ) == 0:
                reduce >>= 1
            just_3.append((val, reduce%8))
        mod = val % 8
        if mod in [0,2,4,6]:
            if mod not in D:
                D[mod] = {}
            if count not in D[mod]:
                D[mod][count] = 0
            D[mod][count] = D[mod][count] + 1
    #
    for label in generationLabels(a):
        val = mrTupValue(mrTupFromPath(label))
        if val[1] == 1:
            collectCount(val[0], countTwos(val[0]))
    return distributionAverage(D), D, just_3
#

"""
The following functions implement a rapidly converging twist on the Collatz Conjecture. 
When an initial number is chosen that forces a large number of diverging steps, 
it does not seem to matter how large the initial number of diverging steps is
and the algorithm recovers begins converging after two steps, recovers to approximately the 
initial log(n, 2) within 5 steps and then rapidly converges to one.

Will this always recover in 5 steps?  Can a bound on worse case converging times be
found for this form of the problem?
"""
def collatzPath(collatzNumber):
    path = []
    while collatzNumber != 1:
        if (collatzNumber & 1) == 0:
            collatzNumber = collatzNumber // 2
            path.append("1")
        else:
            collatzNumber = (3 * collatzNumber + 1) // 2
            path.append("0")
    return "".join(path)
#
def computeNextPrefixBit(a, p2, label, mod):
    # We cannot choose two label bits, so we generate an exemplar
    example = p2 + mod 
    label = collatzPath(example) 
    return a+1, 2**(a+1), label[0:a]
    
def computePrefixForModClass(power_of_2, n):
    mod = n % 4
    label = ["11", "01", "10", "00"][mod]
    if power_of_2 > 2:
        a = 3
        p2 = 2**(a)
        while a <= power_of_2:
            a, p2, label = computeNextPrefixBit(a, p2, label, n % p2)
    return label
#

def maxOddPowerOf2(n):
    p2 = math.floor(math.log(n, 2.0))
    if p2 & 1 != 1:
        p2 -=1
    if p2 < 3:
        p2 = 3
    return p2
#
def affineFunctionParamsFromPrefix(prefix):
    """
    Return the affine parameters A, B for n' = A*n + B

    (Logic extracted from earlier mapByPrefix function)
    """
    a = len(prefix)
    b = countZeros(prefix)
    accum = 0
    sgn = -1**(a)
    for i in range(a-1):
        sgn = sgn * -1
        P = [2]*(a) # we do not do the zz term
        if prefix[i] == "0":
            P[i] = 1
        else:
            continue  # y = 0, so product will be 0
        for j in range(i+1, a, 1):
            if prefix[j] == "0":
                P[j] = -3
            else:
                P[j] = -1
        #print((sgn, P))
        # Trying to avoid an overflow condition ...
        if sgn == 1:
            accum = accum + np.prod(P)
        else:
            accum = accum - np.prod(P)

    if prefix[-1] == "0":
        #print((-sgn, [2]*(a-1)))
        accum -= (sgn * (2**(a-1)))
    #print(b, accum)
    return(3**b, int(accum)) # get rid of np.int before returning
#
def acceleratedConvergeOdd(n):
    p2 = maxOddPowerOf2(n)  # Get the largest odd power of 2 less than n
    """
    Identify and apply the maximum affine mapping to
    move the odd parameter to the 0(mod 2^{p2}) portion of the lattice
    """
    prefix = computePrefixForModClass(p2, n)
    A, B = affineFunctionParamsFromPrefix(prefix)
    n_ = A*n + B
    return n_
#
def acceleratedConvergeEven(n):
    """
    Remove all powers of two from n to get the next odd number
    """
    n_ = n
    while n_ & 1 == 0:
        n_ >>=1
    return n_
#

def collatzPath(collatzNumber):
    path = []
    while collatzNumber != 1:
        if (collatzNumber & 1) == 0:
            collatzNumber = collatzNumber // 2
            path.append("1")
        else:
            collatzNumber = (3 * collatzNumber + 1) // 2
            path.append("0")
    return "".join(path)
#
def computeNextPrefixBit(a, p2, label, mod):
    # We cannot choose two label bits, so we generate an exemplar
    example = p2 + mod 
    label = collatzPath(example) 
    return a+1, 2**(a+1), label[0:a]
    
def computePrefixForModClass(power_of_2, n):
    mod = n % 4
    label = ["11", "01", "10", "00"][mod]
    if power_of_2 > 2:
        a = 3
        p2 = 2**(a)
        while a <= power_of_2:
            a, p2, label = computeNextPrefixBit(a, p2, label, n % p2)
    return label
#

def prefixForModClass(power_of_2, n):
    mod = n % 4
    prefix = ["11", "01", "10", "00"][mod]
    if power_of_2 > 2:
        a = 3
        while a <= power_of_2:
            p2 = 2**(a)
            mod = n % p2
            exemplar = p2 + mod
            bits = []
            while exemplar != 1:
                if (exemplar & 1) == 0:
                    bits.append("1")
                    exemplar = exemplar // 2
                else:
                    bits.append("0")
                    exemplar = (3 * exemplar + 1) // 2
            next_bit = bits[a-1]
            prefix = prefix + next_bit
            a = a+1
    return prefix
#

def findFirstLowerMods():
    mod_list = []
    mi = 0
    found = []
    D = {}
    depth = 8
    prefix = "000"
    for i in range(8):
        # We want to find the first instance in the lattice of these moduli
        mod_list.append( (2**(3*i)-1,  2**(3*i)) )
    #
    while len(D) != len(mod_list):
        for suffix in generationLabels(depth):
            label = prefix + suffix
            val = mrTupValue(mrTupFromPath(label))
            if val[1] == 1:
                if val[0] % mod_list[mi][1] == mod_list[mi][0]:
                    if val[0] not in found:
                        print(f"found {val}, {label} for {mod_list[mi]}")
                        found.append(val[0])
                        D[mod_list[mi]] = (val, label)
                        mi += 1
        depth += 3
        prefix += "00"
    #
    return D
#

def bottom_val(a):
    """
    OEIS A002450

    Not really the "bottom value", more like "top of bottom 1/4" for even generations and "top of bottom half" for odd generations

    Has 1:1 mapping via applyMod8Affine to 2^n
    """
    if a < 2:
        return None
    if a & 1 == 1:
        return 2*bottom_val(a-1)
    j = a//2 
    val = 0
    for i in range(j):
        val <<=2
        val |= 1
    return val
#

def affineFunctionParamsFromPrefix(prefix):
    """
    Return the affine parameters A, B for n' = A*n + B

    (Logic extracted from earlier mapByPrefix function)
    """
    a = len(prefix)
    b = countZeros(prefix)
    accum = 0
    sgn = -1**(a)
    for i in range(a-1):
        sgn = sgn * -1
        P = [2]*(a) 
        if prefix[i] == "0":
            P[i] = 1
        else:
            continue  # y = 0, so product will be 0
        for j in range(i+1, a, 1):
            if prefix[j] == "0":
                P[j] = -3
            else:
                P[j] = -1
        #print((sgn, P))
        accum += (sgn * np.prod(P))

    if prefix[-1] == "0":
        #print((-sgn, [2]*(a-1)))
        accum -= (sgn * (2**(a-1)))
    #print(b, accum)
    return(3**b, int(accum)) # get rid of np.int before returning
#

def efficient_binary_arrangements(num_zeros, num_ones):
    """
    Generates all unique arrangements of a string with a given number 
    of black and white beads efficiently using combinations.
    """
    total_length = num_zeros + num_ones
    # Define bead types
    black = '0'
    white = '1'

    # Get all combinations of indices where black beads will be placed
    # This returns an iterator of tuples of indices
    black_indices_combinations = combinations(range(total_length), num_zeros)
    
    unique_arrangements = []
    
    # Iterate directly over the unique combinations
    for black_indices in black_indices_combinations:
        # Create a list representation of the current arrangement
        arrangement_list = [white] * total_length
        for i in black_indices:
            arrangement_list[i] = black

        yield "".join(arrangement_list)
#

def direct_0(a):
    """
    This function directly generates all integers of the given generation $a$
    that have one zero in their label
    """
    if a < 4:
        return []
    #
    collatzNums = []
    if (a & 1) == 0:
        i_0 = 0
    else:
        i_0 = 1
    for c0 in range(i_0, a-3, 2):
        n = (2**a  - 2**c0)//3
        collatzNums.append(n)
    return collatzNums
#
def directTup_00(a):
    """
    This function directly generates all mrTyps of the given generation $a$
    that have two zeros in their label
    """
    def swapParity(parity_):
        return parity_ ^ 1
    #
    
    b = 2
    c1 = a-4
    delta_c01 = [1, 2]
    delta_c0 = [4, 2]
    parity = 0
    while c1 >= 1:
        c0 = c1 - delta_c01[parity]
        while c0 >= 0:
            n = (2**a - 3*(2**c0) - (2**c1))//9
            yield ((a, -b), ((c0, -1), (c1, -2)))
            c0 -= 2
        #
        c1 -= delta_c0[parity]
        parity = swapParity(parity)
    #
#
# The double-zero numbers directly:
def direct_00(a):
    """
    This function directly generates all integers of the given generation $a$
    that have two zeros in their label

    This function give the same answer as brute-force methods such as ZeroSumSet approach
    for large generations (27, 32, 41, ...
    """
    collatzNums = []
    
    def swapParity(parity_):
        return parity_ ^ 1
    #
    
    b = 2
    c1 = a-4
    delta_c01 = [1, 2]
    delta_c0 = [4, 2]
    parity = 0
    while c1 >= 1:
        c0 = c1 - delta_c01[parity]
        while c0 >= 0:
            n = (2**a - 3*(2**c0) - (2**c1))//9
            collatzNums.append(n)
            c0 -= 2
        #
        c1 -= delta_c0[parity]
        parity = swapParity(parity)
    #
    return collatzNums
#

def generationModulusChoices(a, b):
    """
    Given a,b with the "standard" meanings:
    a: generation, power of 2 of leading term
    b: number of zeros, power of 3 of denominator

    This function generates all possible 0 index positions for an integer result by testing
    each possibility against the modulus math.

    This is actually more brute force than directly computing the Tuple value and checking for
    a denominator of 1.

    The point is to better understand the structural lattice patterns that lead to integers.
    
    """    
    debug_counter = 0
    denom = 3**b
    leading_term = 2**a % denom
    if a <= 3:
        yield None
    # Added this cache partly to help understand what we are iterating over
    cache = {}
    def get_modulus(c, d):
        if d not in cache:
            cache[d] = {}
        if c not in cache[d]:
            mod = denom - ((((3**(d)) * (2**c))) % denom)
            cache[d][c] = mod
        else:
            mod = cache[d][c]
        return mod
    #
            
    values = range(0, a - 3)
    for item in combinations(values, b):
        debug_counter += 1
        if (debug_counter % 100000000) == 0:
            print(f"{debug_counter} combinations processed")
        current = [leading_term]
        zero_idxes = []
        for i in range(len(item)):
            current.append( get_modulus(item[i], b-i-1) )
            zero_idxes.append(item[i])
        if sum(current) % denom == 0:
            yield (current, denom, zero_idxes)

#    
def countGenerationModulusChoices(a, b):
    n = 0
    for tup in generationModulusChoices(a, b):
        n += 1
    return n
#

def generationLabelChoices(a, b):
    values = range(0, a)
    for item in combinations(values, b):
        L = ["1"]*a
        for idx in item:
            L[idx] = "0"
        yield "".join(L)
#

"""
Cache all Collatz Paths
"""
D = {}
def encodePath(label):
    """
    Save a little memory by storing labels as Python mega integers
    """
    return (len(label), int(label, 2))
def decodePath(lnlabel):
    return f"{lnlabel[1]:0{lnlabel[0]}b}"
def getLabel(nn):
    if nn not in D:
        label = collatzPath(nn)
        D[nn] = encodePath(label)
    else:
        label = decodePath(D[nn])
    return label
#
def next_A092893(tup):
    """
    Use the a,b of the previous A092893 found to 
    constrain the next value search.
    """
    a, b, n = tup
    b_ = b+1
    for i in range(4):
        a_ = a+i+1
        # For a given $a, b$ this is the upper bound of the integer search space.  This 
        # number is not small, but is MUCH smaller than $2^a$.  If we have not found
        # anything in this generation when we reach this upper bound, then on to the 
        # next generation ($a$).
        upper = math.floor((2**(a_))/(3**b_))
        if upper & 1 == 0:
            upper -= 1
        for n_ in range(3, upper, 2):
            label = getLabel(n_)
            if countZeros(label) == b_:
                return (a_, b_, n_)            
    return None
#
def next_smallest_generation_value(tup):
    """
    We want to find the smallest value that uses same number of zeros for a given generation as the above function.
    """
    """
    Use the a,b of the previous A092893 found to 
    constrain the next value search.
    """
    a, b, n, m, label = tup
    b_ = b+1
    for i in range(4):
        a_ = a+i+1
        # For a given $a, b$ this is the upper bound of the integer search space.  This 
        # number is not small, but is MUCH smaller than $2^a$.  If we have not found
        # anything in this generation when we reach this upper bound, then on to the 
        # next generation ($a$).
        upper = math.floor((2**(a_))/(3**b_))
        if upper & 1 == 0:
            upper -= 1
        for n_ in range(3, upper, 2):
            label = getLabel(n_)
            if countZeros(label) == b_:
                break
    print(f"found b={b} for a={a}")

    # Now find the value that has the largest subractend and therefor the lowest value for a_, b_
    min_val = 1
    for prefix in efficient_binary_arrangements(b_, a_-3):
        # Flip prefix so we process right-most zeros first
        label_ = prefix[::-1] + "111"
        T = mrTupFromPath(label_)
        val = mrTupValue(T)
        if val[1] == 1:
            ## The reversal of prefix always gives us the smallest number first
            min_val = val[0]
            break
        
    if min_val != 1:
        return (a_, b_, n_, min_val, label_)
    else:
        None
    
#

def next_smallest_generation_odd_value(tup):
    """
    We want to find the smallest odd value that uses same number of zeros for a given generation as the above function.
    """
    """
    Use the a,b of the previous A092893 found to 
    constrain the next value search.
    """
    a, b, n, m, label = tup
    b_ = b+1
    for i in range(4):
        a_ = a+i+1
        # For a given $a, b$ this is the upper bound of the integer search space.  This 
        # number is not small, but is MUCH smaller than $2^a$.  If we have not found
        # anything in this generation when we reach this upper bound, then on to the 
        # next generation ($a$).
        upper = math.floor((2**(a_))/(3**b_))
        if upper & 1 == 0:
            upper -= 1
        for n_ in range(3, upper, 2):
            label = getLabel(n_)
            if countZeros(label) == b_:
                break

    # Now find the value that has the largest subractend and therefor the lowest value for a_, b_
    min_val = 1
    for prefix in efficient_binary_arrangements(b_-1, a_-4):
        # Flip prefix so we process right-most zeros first
        label_ = "0" + prefix[::-1] + "111"
        T = mrTupFromPath(label_)
        val = mrTupValue(T)
        if val[1] == 1:
            ## The reversal of prefix always gives us the smallest number first
            min_val = val[0]
            break
        
    if min_val != 1:
        return (a_, b_, n_, min_val, label_)
    else:
        None
    
#

#
#  Per-Integer Affine Transform Code
#

def mrTupToGenerationAffineTup(T):
    a_b = T[0]
    a_b_ = (a_b[0], -a_b[1])
    L = T[1]
    L_ = []
    for c_d in L:
        c_d_ = (c_d[0], a_b_[1] + c_d[1])
        L_.append(c_d_)
    AT = (a_b_, L_)
    return AT
#
def mrTupToGenerationAffineParams(T):
    AT = mrTupToGenerationAffineTup(T)
    A = 3**(AT[0][1])
    B = 0
    for c_d in AT[1]:
        B += (2**c_d[0])*(3**c_d[1])
    return (A, B)
#
def generationAffineTupFromPath(label):
    """ Create an  given a path (label)
    """
    mr_tup = N_i
    for bit in label:
        if bit == "1":
            mr_tup = F_1(mr_tup)
        else:
            mr_tup = F_0(mr_tup)
    return mrTupToGenerationAffineTup(mr_tup)
#
def generationAffineParamsFromPath(label):
    AT = generationAffineTupFromPath(label)
    A = 3**(AT[0][1])
    B = 0
    for c_d in AT[1]:
        B += (2**c_d[0])*(3**c_d[1])
    return (A, B)
#

# This is what was ultimately used to create Pick Matrices
def generationAffineParamsFromPath(label):
    """
    Generate the Affine Parameters for Ax + B that converts any label in the lattice to 
    the power of 2 of its generation.

    e.g.  5 -> "0111" -> A=3,B=1 -> 3*5 + 1 -> 2**4
    """
    a, b = len(label), countZeros(label)
    A = 3**(b)
    d = b - 1
    B = 0
    for c in range(a):
        if label[c] == "0":
            B += (2**c)*(3**d) 
            d -= 1
    return A, B
#
def generationIntegersForwardWithMeta_(a, previous):
    """
    Supporting function for next method, generates next generation from the previous
    """
    for n_a_b_l in previous:
        # Ignore the $a$ in the tuple
        n, _, b, label = n_a_b_l
        label_ = "1" + label
        yield (2 * n, a, b, label_)
        numer = 2*n - 1
        if numer % 3 == 0:
            n_ = numer // 3
            label_ = "0" + label
            # Skip degenerate values:
            if label_[-2:] != "01":
                yield (n_, a, b+1, label_)
#
def generationIntegersForwardWithMeta(a):
    """
    Quickly generate all integers in the Collatz tree out to the given generation
    """
    # The starting point
    previous = [(1, 0, 0, "")]
    yield previous[0]
    
    for a_ in range(1, a+1):
        accum = []
        for tup_yielded in generationIntegersForwardWithMeta_(a_, previous):
            accum.append(tup_yielded)
            yield tup_yielded
        previous = accum
#
def write_generationIntegersForwardWithMeta(filename, a):
    """
    Save all Collatz integers up to a given generation $a$ out to a file
    """
    with open(filename, "w") as g:
        for tup in generationIntegersForwardWithMeta(a):
            line = "\t".join(map(str, [n for n in tup])) + "\n"
            g.write(line)
    #
#
