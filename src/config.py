import os
import configparser

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, 'config.ini')

# --- LOAD CONFIGURATION FROM .INI FILE ---
config = configparser.ConfigParser()

# We check if the file exists to avoid confusing errors
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Configuration file not found at: {CONFIG_FILE}")

config.read(CONFIG_FILE, encoding='utf-8')

# --- PATHS FROM CONFIG ---
try:
    paths_section = config['PATHS']
    DATA_DIR = os.path.join(BASE_DIR, os.path.dirname(paths_section.get('corpus_dir', 'data/corpus')))
    CORPUS_DATA_DIR = os.path.join(BASE_DIR, paths_section.get('corpus_dir', 'data/corpus'))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, paths_section.get('processed_dir', 'data/processed'))
    TFIDF_JSON_PATH = os.path.join(BASE_DIR, paths_section.get('tfidf_json', 'data/processed/tfidf.json'))
except KeyError as e:
    raise KeyError(f"Missing PATHS configuration key in config.ini: {e}")

# --- NLP SETTINGS ---
# We convert strings from the .ini to the correct types (int, boolean)
try:
    MIN_WORD_LENGTH = int(config['NLP']['min_word_length'])
    CHARS_TO_REMOVE_REGEX = config['NLP']['chars_to_remove_regex']
    LANGUAGE = config['NLP']['language']
except KeyError as e:
    raise KeyError(f"Missing configuration key in config.ini: {e}")