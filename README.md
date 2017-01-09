# Exercises on neural networks
This repository contains some fun exercises we did on neural networks.

We hope this repository could be helpful for people with similar interests who want to find clear and concise examples about TensorFlow.

## PTB language modeling using RNNs

This re-implements [TensorFlow's RNN language modeling tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb) in a more object-oriented fashion. It supports various RNN cell types, as well as bi-directional RNN.

Example command to start model training from scratch:

```ShellSession
python ptb_model_trainer.py --data_directory=./data/ptb_data/ --cell_type=LSTM --base_lr=0.01 --num_units=100 --num_epochs=100 --batch_size=20 --num_steps=35 --vocab_size=10000 --bidirectional=False
```

Example command to restore model training from a previously saved checkpoint:

```ShellSession
python ptb_model_trainer.py --data_directory=./data/ptb_data/ --cell_type=LSTM --base_lr=0.01 --num_units=100 --num_epochs=100 --batch_size=20 --num_steps=35 --vocab_size=10000 --bidirectional=False --model_id=-500 --run_name=1484638602/
```

### Experimental results
