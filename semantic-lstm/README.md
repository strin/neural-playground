# Conditionally Semantic LSTM for Dialogue Modeling

## Dataset

A dataset is (dialogue act, utterance) pairs.

| Dataset Name  | Size      | 
| ------------- |:---------:|
| SF Restaurant | 6196      |
| SF Hotel      | 6384      |



## SC-LSTM

Implementation of semantically conditioned LSTM.

![](https://www.dropbox.com/s/fozxom37fizhg4e/Screenshot%202016-08-09%2010.34.00.png?dl=1)

### Using pretrained wordvec

Reference:
[GloVe: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf)

Download pretrained wordvec models from the [github page](https://github.com/stanfordnlp/GloVe).

### Delexicalize

To allow better generalization across different slot values, we first de-lexicalize the values in the input sentence, for example.

```
{'__type__': 'hello'}
hello , welcome to the parlance dialogue system ? you can ask for restaurant -s by area , price range or food type . how may i help you ?

{'price_range': 'moderate', '__type__': 'inform', 'name': "'trattoria contadina'"}
<name> is a nice restaurant in the moderate price range .

{'price_range': 'moderate', '__type__': 'inform', 'name': "'alamo square seafood grill'", 'area': "'friendship village'"}
<name> is a nice restaurant in the area of <area> and it is in the moderate price range .

{'__type__': 'inform', 'name': "'alamo square seafood grill'", 'address': "'803 fillmore street'"}
the address for <name> is <address> .
```

**Discussion:** 1-hot vector representation vs. generalizable representation.


### Minor Todos

- [ ] Init for the DA embedding layer.
- [ ] Pretrain word vectors.
- [ ] consume less optimization.
- [ ] zero padding to sentences and the efficiency.
- [ ] theano compute shape information at run time.
- [ ] difference between hidden states h and c.
- [ ] fix words at each time step during training time.