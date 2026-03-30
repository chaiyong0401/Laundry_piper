import os

def count_json_files():
    current_dir = os.getcwd()
    files = os.listdir(current_dir)
    json_files = [f for f in files if f.endswith('.json') and os.path.isfile(f)]
    return len(json_files)

if __name__ == "__main__":
    count = count_json_files()
    print(f"Number of .json files: {count}")
