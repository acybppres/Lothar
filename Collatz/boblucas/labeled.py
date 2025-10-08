


def necklaces(N, K):
    result = []
    
    # This can be optimized by checking symbol for symbol and early exiting, but this is slightly clearer
    def is_representative(s, periods):
        for j in periods:
            rotated = s[j:] + s[:j]
            # The best transposition is the one that makes the first symbol 0 at the current rotation
            transposed = [(x-rotated[0]+K)%K for x in rotated]
            if transposed < s:
                return False
        return True

    def necklaces_(s, periods = []):
        # Terminating condition, when N symbols are placed
        if len(s) == N:
            # When the period divides the string length we are done, otherwise check
            if is_representative(s, periods):
                result.append(tuple(s))
            return
     
        # Create larger symbol that any of the primitive periods suggest
        # This way we can never create a string that is smaller under string rotation and alphabet rotation
        for i in range(0, K):
            if any((i - s[p])%K < s[-p] for p in periods):
                continue
     
            active_periods = [len(s)] + [p for p in periods if (i - s[p]) % K == s[-p]]
            s.append(i)
            necklaces_(s, active_periods)
            s.pop()
    
    necklaces_([0])
    
    return result
#
