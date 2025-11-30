import os 
import traceback
from utils.color import colored_msg

try:
    MASTER_RANK = int(os.environ['RANK']) == 0
except:
    MASTER_RANK = True

def master_only(func):
    """rank 0에서만 실행하는 데코레이터"""
    def wrapper(*args, **kwargs):
        if MASTER_RANK:
            return func(*args, **kwargs)
    return wrapper

@master_only
def printE(message):
    print(f" {colored_msg('[ERROR]', 'red')} {message}\n")
    traceback.print_exc()

@master_only
def printS(message):
    print(f" {colored_msg('[SYSTEMS]', 'blue')} {message}")

master_only
def printW(message):
    print(f" {colored_msg('[WARNING]', 'yellow')} {message}")

@master_only
def printT(message):
    print(f" {colored_msg('[TRAIN]', 'green')} {message}")

@master_only
def printM(message, color=None):
    print(colored_msg(message, color))

def line(func):
    def wrapper(*args, **kwargs):
        printM("\n==================================================================\n")
        func(*args, **kwargs)  
        printM("\n==================================================================\n")
    return wrapper    

def time_check(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        printM(f"Model inference for {end_time - start_time:.4f} seconds", end='\r')
        return result
    return wrapper

