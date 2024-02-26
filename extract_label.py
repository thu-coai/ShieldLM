import argparse
import json
from tqdm import tqdm

def extract_label(text, lang):
    text = text.strip()
    if lang == 'en':
        if '[Answer] unsafe' in text:
            label = 1
        elif '[Answer] safe' in text:
            label = 0
        elif '[Answer] controversial' in text:
            label = 2
        else:
            raise ValueError(f'Failed to extract label from text: {text}')
        
    if lang == 'zh':
        if '[答案] 不安全' in text:
            label = 1
        elif '[答案] 安全' in text:
            label = 0
        elif '[答案] 有争议' in text:
            label = 2
        else:
            raise ValueError(f'Failed to extract label from text: {text}')
    
    return label
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=None, type=str, help='path of input file in a jsonl format')
    parser.add_argument('--output_path', default=None, type=str, help='path of output file in a jsonl format')
    parser.add_argument('--lang', default=None, type=str, required=True, choices=('en', 'zh'))

    args = parser.parse_args()
    with open(args.input_path, 'r') as f:
        datas = []
        for line in f:
            datas.append(json.loads(line))
    
    for d in tqdm(datas):
        d['label'] = extract_label(d['output'], args.lang)
        
    with open(args.output_path, 'w') as f:
        for d in datas:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    