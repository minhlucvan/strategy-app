
import toml
import os 

file_path = '.streamlit/secrets.toml'

# ensure the file exists
if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        f.write('')

def load_config():
    if not os.path.exists(file_path):
        return {}
    
    return toml.load(file_path)

config_dict = None

def refresh_config():
    global config_dict
    config_dict = load_config()


def update_config(config_dict: dict):
    secrets_dict = load_config()

    for key in config_dict:
        secrets_dict[key] = config_dict[key]
        
    with open(file_path, 'w') as f:
        toml.dump(secrets_dict, f)

    refresh_config()

def deep_value(dictionary, key):
    keys = key.split('.')

    for key in keys:
        if isinstance(dictionary, dict):
            dictionary = dictionary.get(key, None)
        else:
            return None
    return dictionary


def get_config(key: str):
    return deep_value(config_dict, key)

def clear_cache():
    # remove ./cache/demo_cache.sqlite
    if os.path.exists('./cache/demo_cache.sqlite'):
        os.remove('./cache/demo_cache.sqlite')
        
        
refresh_config()