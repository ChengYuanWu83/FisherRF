import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in: {execution_time:.9f} seconds")
        return result
    return wrapper

@time_function
def sample_function():

    print("Function execution completed.")

# 調用範例函式
sample_function()
