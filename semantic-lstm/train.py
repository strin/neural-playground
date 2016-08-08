# train a semantically conditioned LSTM.
import json
from pprint import pprint
import re
import traceback

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


def parse_dact(raw_da):
    da = {}
    try:
        result = re.findall(r'(.*)\((.*)\)', raw_da)[0]
        da['type'] = result[0]
        da['args'] = []
        for raw_field in result[1].split(';'):
            if not raw_field:
                continue
            if '=' in raw_field:
                field = list(re.findall(r'(.*)=(.*)', raw_field)[0])
                da[field[0]] = field[1]
            else:
                field = raw_field
                da['args'].append(field)

    except Exception as e:
        print '[error]', e.message
        print 'raw_da', raw_da
        traceback.print_exc()
        exit(0)
    return da


if __name__ == '__main__':
    data = read_data('data/dact/%s/train+valid+test.json' % DOMAIN)
    for d in data:
        pprint(parse_dact(d['dact']))
