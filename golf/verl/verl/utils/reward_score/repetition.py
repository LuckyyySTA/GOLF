import re

def detect_repetition_with_hash(text, window_size=10, max_repetitions_limit=6):
    """
    Use hashing to efficiently detect repeated n-grams (split by space and underscore).
    Returns -1 if any specific n-gram repeats more than 6 times, otherwise 0.
    """
    # Split text by both space and underscore
    words = []
    for segment in text.split():
        words.extend(segment.split('_'))
    
    if len(words) <= window_size:
        return 0
    
    hash_counts = {}
    max_repetitions = 0
    
    for i in range(len(words) - window_size + 1):
        # Get window and its hash
        window = tuple(words[i:i+window_size])
        window_hash = hash(window)
        
        # Update count for this hash
        hash_counts[window_hash] = hash_counts.get(window_hash, 0) + 1
        
        # Update max repetitions and early exit if threshold crossed
        if hash_counts[window_hash] > max_repetitions:
            max_repetitions = hash_counts[window_hash]
            if max_repetitions >= max_repetitions_limit:
                return -1
    
    return 0
