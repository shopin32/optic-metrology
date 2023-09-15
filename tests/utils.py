import os;

def get_test_file(file_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'testdata', file_name)