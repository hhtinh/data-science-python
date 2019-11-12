import re

def validate(username):
    is_valid = False
    if re.findall("^(?=.{6,16}$)(?![_])[a-zA-Z0-9._][\S]+(?<![-])$", username):
        is_valid = True
    if is_valid:
        return f"{username} is valid"
    else:
        return f"{username} is NOT valid"
    
print(validate("Mike-Standish-a-very-long")) #Invalid username
print(validate("Mike Standish")) #Invalid username
print(validate("Mike-Standish-")) #Invalid username
