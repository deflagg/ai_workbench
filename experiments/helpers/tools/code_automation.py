from langchain_experimental.utilities import PythonREPL
from typing_extensions import  Annotated
from langchain_core.tools import tool


# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()
repl.globals = globals()
repl.locals = locals()

@tool
def python_repl(
    code: Annotated[str, "The raw valid python code."],
):
    """
    Use this to execute python commands. Input should be a valid python command. 
    If you want to see the output of a value, you should print it out with `print(...)`.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"CODE:\n{code}\nFUNCTION OUTPUT:\n{result}"
    
    return result_str


# input_code = """
# def is_prime(n):
#     if n <= 1:
#         return False
#     for i in range(2, int(n**0.5) + 1):
#         if n % i == 0:
#             return False
#     return True

# test_cases = [1, 2, 3, 4, 5, 16, 17, 18, 19, 20]
# results = {n: is_prime(n) for n in test_cases}
# print(results)"""

# input_code = """
# def square(x):
#     return x ** 2

# result = {key: square(key) for key in range(5)}
# print(result)
# """

# import sys
# from io import StringIO

# old_stdout = sys.stdout
# sys.stdout = mystdout = StringIO()

# cleaned_command = repl.sanitize_input(input_code)
# ret = exec(cleaned_command, globals(), locals())
# sys.stdout = old_stdout
     
# test = 'def square(x):\n    return x ** 2\n\nresult = {key: square(key) for key in range(5)}\nprint(result)'

# ret = exec(test)
# # ret = repl.run('def square(x):\n    return x ** 2\n\nresult = {key: square(key) for key in range(5)}\nprint(result)')
# ret = python_repl(input_code)
# print(ret)
