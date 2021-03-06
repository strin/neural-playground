# train a semantically conditioned LSTM.
from __init__ import *
from layer import SemConLSTM
import dill as pickle

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
        da['__type__'] = result[0]
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
        d['dact_int'] = d_new
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


def build(da_vocab, word_vocab, max_senlen, word_emb=None, hidden_dim=300):
    '''
    build sc-lstm model.
    '''
    model_da = Sequential()
    # da embedding layer.
    model_da.add(Embedding(len(da_vocab), output_dim=hidden_dim))
    # bag-of-words averaging.
    model_da.add(Lambda(lambda emb: K.sum(emb, axis=1), output_shape=lambda input_shape: (input_shape[0], input_shape[2])))
    model_da.add(RepeatVector(max_senlen))
    # word embedding layer.
    model_word = Sequential()
    model_word.add(Embedding(len(word_vocab), input_length=max_senlen,
                             output_dim=hidden_dim, weights=[word_emb])
                   )
    model_word.add(Reshape((max_senlen, hidden_dim)))
    # sc-lstm layer.
    model = Sequential()
    model.add(Merge([model_word, model_da], mode='concat', concat_axis=2))
    model.add(SemConLSTM(hidden_dim, da_dim=hidden_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(len(word_vocab), activation='softmax')))
    # loss function.
    def loss(y_true, y_pred):
        print y_true
        print y_pred
        return K.sparse_categorical_crossentropy(y_pred[:, :max_senlen-1, :],
                                                 y_true[:, 1:, :])

    model.compile('rmsprop', 'sparse_categorical_crossentropy')

    # run test.
    lets_test = False
    if lets_test:
        model_da.compile('rmsprop', 'mse') # arbitrary
        model_word.compile('rmsprop', 'mse')
        model.compile('rmsprop', 'mse')
        model_da.predict(np.zeros((32, 10), dtype=np.int64))
        model_word.predict(np.zeros((32, max_senlen), dtype=np.int64))
        model.predict([np.zeros((32, max_senlen), dtype=np.int64),
                       np.zeros((32, 10), dtype=np.int64)])
    return model


def build_vocab(data, output_file=None):
    '''
    will delexicalize based on dialect act.
    '''
    vocab = {}
    log = open('log/delexicalize_%s' % DOMAIN, 'w')
    for d in data:
        sen = d['ref']
        # delexicalize.
        print>>log, d['dact']
        for token, value in d['dact'].items():
            if token in ['name', 'type', 'price', 'phone', 'count',
                         'address', 'postcode', 'area', 'near']:
                value = value.replace('\'', '')
                sen = sen.replace(value, '<' + token + '>')
        print>>log, sen, '\n'
        words = sen.replace('\n', '').strip().split(' ')
        words = [word for word in words if word]
        words = ['<bos>'] + words + ['<eos>']
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
        d['sen'] = [vocab[word] for word in words]
    log.close()
    if output_file:
        with open(output_file, 'w') as f:
            for word in vocab:
                f.write(word + '\n')
    return vocab


def load_wordvec(vocab, wordvec_path):
    with open(wordvec_path, 'r') as f:
        wordvec = pickle.load(f)
    embedding = []
    for (i, word) in enumerate(vocab):
        if word.startswith('<') and word.endswith('>'):
            emb = np.zeros_like(wordvec['hi'])
        elif word not in wordvec:
            emb = wordvec['<unk>']
        else:
            emb = wordvec[word]
        embedding.append(emb)
    return np.array(embedding, dtype=np.float32)


def create_X_Y(data):
    xs = []
    ys = []
    das = []
    for d in data:
        xs.append(d['sv'])
        ys.append(d['sv'])
        das.append(d['dv'])
    xs = np.array(xs, dtype=np.int64)
    ys = np.array(ys, dtype=np.int64)
    ys = ys.reshape(ys.shape + (1,))
    das = np.array(das, dtype=np.int64)
    return (xs, ys, das)



if __name__ == '__main__':
    data = read_data('data/dact/%s/train+valid+test.json' % DOMAIN)
    for d in data:
        d['raw_dact'] = str(d['dact'])
        d['dact'] = parse_dact(d['raw_dact'])
    (data_encoded, da_vocab) = prepare_dact(data)
    #word_vocab = {i:i for i in range(100)}
    #build(da_vocab, word_vocab, max_senlen=10)
    #build_vocab(data, 'vocab_sfxrestaurant.txt')
    vocab = build_vocab(data)
    max_senlen = 0
    max_dactlen = 0
    for d in data:
        if len(d['sen']) > max_senlen:
            max_senlen = len(d['sen'])
        if len(d['dact_int']) > max_dactlen:
            max_dactlen = len(d['dact_int'])
    print 'vocab size', len(vocab)
    print 'max sentence len', max_senlen
    print 'max dialogue act len', max_dactlen
    for d in data:
        d['sv'] = np.zeros(max_senlen, dtype=np.int64)
        d['sv'][:len(d['sen'])] = d['sen']
        d['dv'] = np.zeros(max_dactlen, dtype=np.int64)
        d['dv'][:len(d['dact_int'])] = d['dact_int']
    word_emb = load_wordvec(vocab, 'wordvec_%s.pkl' % DOMAIN)
    (xs, ys, das) = create_X_Y(data)
    model = build(da_vocab, vocab, max_senlen, word_emb,
          hidden_dim=word_emb.shape[1])
    with open('result/model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('result/model-init.weights')

    for ei in range(10):
        model.fit([xs, das], ys, batch_size=32, nb_epoch=1)
        model.save_weights('result/model-%d.weights' % ei)
