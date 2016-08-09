# train a semantically conditioned LSTM.
from __init__ import *
from layer import SemConLSTM

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
    for x in encoded:
        pair = ivocab[x].split('.')
        dact[pair[0]] = pair[1]
    return dact


def build(da_vocab, word_vocab, max_senlen,
          DA_EMBED_DIM=100, WORD_EMBED_DIM=300,
          LSTM_DIM=300):
    '''
    build sc-lstm model.
    '''
    model_da = Sequential()
    # da embedding layer.
    model_da.add(Embedding(len(da_vocab), output_dim=DA_EMBED_DIM))
    # bag-of-words averaging.
    model_da.add(Lambda(lambda emb: K.sum(emb, axis=1), output_shape=lambda input_shape: (input_shape[0], input_shape[2])))
    model_da.add(RepeatVector(max_senlen))
    # word embedding layer.
    model_word = Sequential()
    model_word.add(Embedding(len(word_vocab), input_length=max_senlen, output_dim=WORD_EMBED_DIM))
    model_word.add(Reshape((max_senlen, WORD_EMBED_DIM)))
    # sc-lstm layer.
    model = Sequential()
    model.add(Merge([model_word, model_da], mode='concat', concat_axis=2))
    model.add(SemConLSTM(LSTM_DIM, da_dim=DA_EMBED_DIM))
    # run test.
    lets_test = True
    if lets_test:
        model_da.compile('rmsprop', 'mse') # arbitrary
        model_word.compile('rmsprop', 'mse')
        model.compile('rmsprop', 'mse')
        model_da.predict(np.zeros((32, 10), dtype=np.int64))
        model_word.predict(np.zeros((32, max_senlen), dtype=np.int64))
        model.predict([np.zeros((32, max_senlen), dtype=np.int64),
                       np.zeros((32, 10), dtype=np.int64)])






if __name__ == '__main__':
    data = read_data('data/dact/%s/train+valid+test.json' % DOMAIN)
    for d in data:
        d['raw_dact'] = str(d['dact'])
        d['dact'] = parse_dact(d['raw_dact'])
    (data_encoded, da_vocab) = prepare_dact(data)
    word_vocab = {i:i for i in range(100)}
    build(da_vocab, word_vocab, max_senlen=10)
