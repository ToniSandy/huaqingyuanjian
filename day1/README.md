#Python & Git Fundamentals Course
1. Environment preparation
python --version
python -m venv myenv
myenv\Scripts\activate
source myenv/bin/activate
pip install requests

2.Variables, variable types, scopes

- Basic variable types: 'int', 'float', 'str', 'bool', 'list', 'tuple', 'dict', 'set'.
- Scope: Global variables, local variables, 'global' and 'nonlocal' keywords.
- Type conversions: e.g. 'int()', 'str()'.
name = "Alice"  # str
age = 20        # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict
age_str = str(age)
number = int("123")
x = 10 
def my_function():
    y = 5  
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")

3. Operators and Expressions

- Arithmetic operators: ' ', , , '/', '//', '%', '*'.
- Comparison operators: '==', '!=', '>', '<', '>=', '<='.
- Logical operators: 'and', 'or', 'not'.
- Bitwise operators: '&', '|', '^', '<<', '>>'.

a = 10
b = 3
print(a + b)  # 13
print(a // b)  # 3（整除）
print(a ** b)  # 1000（幂）
x = True
y = False
print(x and y)  # False
print(x or y)   # True
print(a > b)  # True
   
4. Statements: Condition, Loop, Exception

- Conditional statements: 'if', 'elif', 'else'.
- Loops: 'for', 'while', 'break', 'continue'.
- Exception handling: 'try', 'except', 'finally'.


score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

for i in range(5):
    if i == 3:
        continue
    print(i)

try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Execution completed.")


6. Packages and Modules: Define Modules, Import Modules, Use Modules, Third-Party Modules

- Module: 'import' statement, 'from ... import ...`。
- Create a module: a '.py' file.
- Packages: Folders containing '__init__.py'.
- Third-party modules: e.g. 'requests', 'numpy'.

def say_hello():
    return "Hello from module!"
import mymodule
print(mymodule.say_hello())
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200
from mypackage import mymodule

7. Classes and Objects

- Class definitions: 'class' keyword, attributes, and methods.
- Inheritance, polymorphism, encapsulation.
- Instantiate objects.


class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())  # I am Alice, 20 years old.
print(grad.introduce())     # I am Bob, a CS student.


8. Decorators
Decorator Essence: A higher-order function that accepts a function and returns a new function.
Use the @ syntax.
Decorator with parameters.


def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hi, {name}!")

greet("Alice")

9. File Operations

- Read and write text files: 'open()', 'read()', 'write()'.
- Context Manager: 'with' statement.
- Process CSV and JSON files.


with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")


with open("example.txt", "r") as f:
    content = f.read()
    print(content)


import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
