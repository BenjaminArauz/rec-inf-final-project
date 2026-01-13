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
try:
    MIN_WORD_LENGTH = int(config['NLP']['min_word_length'])
    LANGUAGE = config['NLP']['language']
    USE_STEMMING = config['NLP'].getboolean('use_stemming', True)
except KeyError as e:
    raise KeyError(f"Missing configuration key in config.ini: {e}")


try:
    MAX_RESULTS = int(config['SEARCH']['max_results'])
except KeyError as e:
    raise KeyError(f"Missing SEARCH configuration key in config.ini: {e}")

# --- CRAWLER SETTINGS ---
try:
    CRAWLER_MAX_PAGES = int(config['CRAWLER']['max_pages'])
    CRAWLER_MAX_DEPTH = int(config['CRAWLER']['max_depth'])
    CRAWLER_DOMAIN_LIMIT = config['CRAWLER'].getboolean('domain_limit', True)
    CRAWLER_TIMEOUT = int(config['CRAWLER']['request_timeout'])
    CRAWLER_KEYWORDS = [kw.strip() for kw in config['CRAWLER'].get('keywords', '').split(',') if kw.strip()]
    CRAWLER_START_URL = config['CRAWLER']['start_url']
except KeyError as e:
    raise KeyError(f"Missing CRAWLER configuration key in config.ini: {e}")