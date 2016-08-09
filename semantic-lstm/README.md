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
### Minor Todos

- [ ] Init for the DA embedding layer.
- [ ] Pretrain word vectors.
- [ ] consume less optimization.
- [ ] zero padding to sentences and the efficiency.
- [ ] theano compute shape information at run time.
- [ ] difference between hidden states h and c.
- [ ] fix words at each time step during training time.