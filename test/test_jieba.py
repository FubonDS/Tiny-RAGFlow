from src.core.tokenizer import JiebaTokenizer
import yaml

config_path = "./config/bm25.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


tokenizer = JiebaTokenizer(
    user_dict_path=config['tokenizer'].get('user_dict_path', None),
    stopwords_path=config['tokenizer'].get('stopwords_path', None),
    normalize_config=config['tokenizer'].get('normalize', {}),
    filter_config=config['tokenizer'].get('filter', {})
)
text = "今天真的很開心😂😂!!! 富邦保險很強🚀🚀"

tokens = tokenizer.tokenize(text)