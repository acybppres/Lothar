# https://github.com/boblucas/necklaces

# Constants for ordinality of strings
EQUAL = 0
TRANSFORMED_LESS = 1
ORIGINAL_LESS = 2

class Mapping:
    UNUSED = -1
    def __init__(self, K, r):
        # The rotation that this instance is mapping, only used for comparison
        self.r = r
        # mapping contains the permutation of the alphabet, s -> t(s)
        self.mapping = [self.UNUSED] * K
        # how often does each characters (from s, not t(s)) occur
        self.references = [0] * K
        # What is the lowest available symbol we can map to
        self.least_open_symbol = 0

    def push(self, x):
        # Map to lowest possible new symbol if we never saw x before
        if self.mapping[x] == self.UNUSED:
            self.mapping[x] = self.least_open_symbol
            self.least_open_symbol += 1

        self.references[x] += 1

    def pop(self, x):
        self.references[x] -= 1
        # If the occurance is 0 free up this symbol in the mapping
        if self.references[x] == 0:
            self.mapping[x] = self.UNUSED
            self.least_open_symbol -= 1

    # Assuming the characters s[r:] were pushed.
    # What is lexiographical ordering of the remapped rotated string versus 's'.
    def compare(self, s):
        if s[-(self.r+1)] > self.mapping[s[-1]]: return TRANSFORMED_LESS
        if s[-(self.r+1)] < self.mapping[s[-1]]: return ORIGINAL_LESS
        return EQUAL

# Keep track of a set of unique numbers
# Remembers operations and allows reverting to a previously marked state
class ActivePeriods:
    def __init__(self):
        self.values = set()
        # List of changes to values
        self.changes = []
        # List of places we should revert to
        self.markings = []

    def begin(self):
        self.markings.append(len(self.changes))

    def add(self, x):
        self.values.add(x)
        self.changes.append(('A', x))

    # Remove any rotations that became invalid
    def remove(self, x):
        self.values.remove(x)
        self.changes.append(('R', x))

    # Undo all removals/additions since last begin call
    def end(self):
        for i in range(0, len(self.changes) - self.markings.pop()):
            x = self.changes.pop()
            if x[0] == 'A':
                self.values.remove(x[1])
            else:
                self.values.add(x[1])
                
# For each rotation of the pushed symbols, keep track of optimal mapping (see above)
# And when a mapping is worse that the original, remove that rotation as relevant
# When pushing a symbol tells you if it will result in lower lex. in any rotation
class Periodicity:
    def __init__(self, N, K):
        self.mappings = [Mapping(K, i) for i in range(0, N)]
        self.active_periods = ActivePeriods()
        self.s = []

    def push(self, x):
        # Add symbol to string
        self.s.append(x)
        # Add new period(eg: rotation)
        self.active_periods.add(len(self.s)-1)
        # Apply symbol to all mappings of all open periods
        for p in self.active_periods.values:
            self.mappings[p].push(x)

        # Before removing all periods that have become invalid, mark this state so that we can move back
        self.active_periods.begin()

        # Check if any active period is lex. lower than the string, if so we stop recursing
        if any(self.mappings[p].compare(self.s) == TRANSFORMED_LESS for p in self.active_periods.values):
            self.pop()
            return False

        # otherwise remove any periods that have become higher lex
        for p in set(self.active_periods.values):
            if self.mappings[p].compare(self.s) == ORIGINAL_LESS:
                self.active_periods.remove(p)

        return True

    def pop(self):
        self.active_periods.end()
        for p in self.active_periods.values:
            self.mappings[p].pop(self.s[-1])

        self.active_periods.remove(len(self.s)-1)
        x = self.s.pop()
        

def necklaces(N, K):
    periods = Periodicity(N, K)
    result = []

    # If we know what rotated (but truncated) strings are equal lex
    # figure out if s is truly representative
    def is_representative(s):
        for p in periods.active_periods.values:
            for i in range(0, p):
                periods.mappings[p].push(s[i])
                a = periods.mappings[p].mapping[s[i]]
                
                if a != s[i-p]:
                    for j in range(i, -1, -1):
                        periods.mappings[p].pop(s[j])
                
                if a < s[i-p]: return False 
                if a > s[i-p]: break
            else:
                for j in range(p-1, -1, -1):
                    periods.mappings[p].pop(s[j])
                
                # Is this string periodic?
                # if not p == 0:
                #     return False
        return True
    #
    # This is a simple DFS that prunes as Periodicity commands
    def necklaces_():
        if len(periods.s) == N:
            if is_representative(periods.s):
                result.append("".join([str(x) for x in periods.s]))
        else:
            for i in range(0, K):
                if periods.push(i):
                    necklaces_()
                    periods.pop()
    #
    necklaces_()
    return result
#


