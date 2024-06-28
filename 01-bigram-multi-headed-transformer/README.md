# Bigram Multi-Headed Transformer Model

Model 01 is a Multi-Headed Transformer model using character tokens

## Features

- **Transformer Architecture**: We use masked attention as per usual
- **Multi-Headed Attention**: The K and V matrices are split up to a moderate number of attention heads
- **Exact Vocabulary**: Prior to first training run the source data is scanned for all unique tokens

## Performance

* Required [massive simplification of character codes in Wikipedia source](https://github.com/jackdanger/wikipedia-parser) to allow 97-char vocab

25%
2 hours in we've processed 11500/45767 batches
training loss: 0.0135

42%
Haven't changed the training loss at all. I can only hope that somehow,
somewhere, floating points numbers are descending a gradient
training loss: 0.0135
