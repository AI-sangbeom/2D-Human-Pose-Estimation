def line(func):
    def wrapper(*args, **kwargs):
        print("\n=============================================\n")
        func(*args, **kwargs)  
        print("\n=============================================\n")
    return wrapper    

def time_check(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Model inference for {end_time - start_time:.4f} seconds", end='\r')
        return result
    return wrapper