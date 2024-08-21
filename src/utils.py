import os 
import string 

def load_prompt_template(file_path):
    with open(file_path, 'r') as file:
        template = file.read()
    return template 


