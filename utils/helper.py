
from typing import Dict
import json

def report(success=True, result=None, error=None):
    summary={
        "success" : success,
        "result" : result,
        "error" : error
    }

    return summary

def create_error(code, message, details, filename, lineno, type=None):
    
    error = {
        "code" : code,
        "message" : message,
        "devel" : {
            "details" : details,
            "filename" : filename,
            "line" : lineno,
            "type" : str(type)
        }
    }

    return error

def read_json(path:str)->Dict:
    """
    Reads and parses a JSON file from the specified file path.

    Args:
        path (str): The file path to the JSON file to be read.

    Returns:
        Dict: A dictionary containing the data parsed from the JSON file.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data
