def pga_signature(b):
    """0(...+)"""
    return 0 if b == 0 else 1

def sta_signature(b):
    """+(...-)"""
    return 1 if b == 0 else -1

def stap_signature(b):
    """0+(...-)"""
    return 0 if b == 0 else (1 if b == 1 else -1)

def cga_signature(b):
    """+-(...+)"""
    return 1 if b == 0 else (-1 if b == 1 else 1)

def positive_signature(b):
    """(...+)"""
    return 1

def negative_signature(b):
    """(...-)"""
    return -1

def null_signature(b):
    """(...0)"""
    return 0