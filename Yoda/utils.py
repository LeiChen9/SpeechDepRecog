import re 

def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True 
    else:
        return False