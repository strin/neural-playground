# train a semantically conditioned LSTM.
import json
from pprint import pprint

DOMAIN = 'sfxrestaurant'

def read_data(path):
    dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith('#')]
        data = json.loads('\n'.join(lines))
        for dial in data:
            for turn in dial['dial']:
                dataset.append(turn['S'])
    return dataset


if __name__ == '__main__':
    data = read_data('data/dact/%s/train+valid+test.json' % DOMAIN)
    print len(data)
