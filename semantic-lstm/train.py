# train a semantically conditioned LSTM.
from __init__ import *

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
        for raw_field in result[1].split(';'):
            if not raw_field:
                continue
            if '=' in raw_field:
                field = list(re.findall(r'(.*)=(.*)', raw_field)[0])
                da[field[0]] = field[1]
            else:
                field = raw_field
                da[field] = 'yes'

    except Exception as e:
        print '[error]', e.message
        print 'raw_da', raw_da
        traceback.print_exc()
        exit(0)
    return da


def prepare_dact(data):
    '''
    encode 'dact' in data with integers.
    return (data_encoded, vocab)
    '''
    vocab = {}
    data_encoded = []
    for d in data:
        d_new = []
        for pair in d['dact'].items():
            pair = list(pair)
            if pair[1].startswith('\''): # string.
                pair[1] = '<str>'
            key = pair[0] + '.' + pair[1]
            if key not in vocab:
                vocab[key] = len(vocab)
            d_new.append(vocab[key])
        data_encoded.append(d_new)
    return (data_encoded, vocab)


def decode_dact(encoded, ivocab):
    '''
    given a list of dact integer encoding, and inverse dict.
    return the dact dict.
    '''
    dact = {}
    print encoded
    for x in encoded:
        pair = ivocab[x].split('.')
        dact[pair[0]] = pair[1]
    return dact


if __name__ == '__main__':
    data = read_data('data/dact/%s/train+valid+test.json' % DOMAIN)
    for d in data:
        d['raw_dact'] = str(d['dact'])
        d['dact'] = parse_dact(d['raw_dact'])
    (data_encoded, vocab) = prepare_dact(data)
    ivocab = create_idict(vocab)
    print data_encoded[:10]
    for i in range(10):
        pprint(decode_dact(data_encoded[i], ivocab))
