# Exercises on neural networks
This repository contains some fun exercises we did on neural networks.

We hope this repository could be helpful for people with similar interests who want to find clear and concise examples about TensorFlow.

## PTB language modeling using RNNs

This re-implements [TensorFlow's RNN language modeling tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb) in a more object-oriented fashion. It supports various RNN cell types, as well as bi-directional RNN.

Similar to the above tutorial, the data required for this tutorial is in the data/ directory of the PTB dataset from Tomas Mikolov's [webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz). I uploaded the word-level data files to this repository.

The file to read and process the PTB data is [copied](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py) from the above tutorial.

Example command to start model training from scratch:

```ShellSession
python ptb_model_trainer.py --data_directory=./data/ptb_data/ --cell_type=LSTM --base_lr=0.01 --num_units=100 --num_epochs=100 --batch_size=20 --num_steps=35 --vocab_size=10000 --bidirectional=False
```

Example command to restore model training from a previously saved checkpoint:

```ShellSession
python ptb_model_trainer.py --data_directory=./data/ptb_data/ --cell_type=LSTM --base_lr=0.01 --num_units=100 --num_epochs=100 --batch_size=20 --num_steps=35 --vocab_size=10000 --bidirectional=False --model_id=-500 --run_name=1484638602
```

### Experimental results
