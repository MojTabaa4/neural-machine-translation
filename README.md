# Attention-based Encoder-Decoder Model for Neural Machine Translation

This repository contains a machine translation model that translates Farsi text to English using an encoder-decoder
architecture.

## Data Preprocessing

The `standardize_en` and `standardize_fa` functions perform text preprocessing. `standardize_en` normalizes text,
converts it to lowercase, keeps only alphabets, space, and selected punctuation marks, removes leading and trailing
whitespaces, and adds special tokens `[START]` and `[END]`. `standardize_fa` removes specific punctuation marks and
performs similar preprocessing as `standardize_en`.

## Example Text

An example Farsi text is defined, and the `standardize_en` function is applied to it to obtain the preprocessed text.

## Regular Expression

Variables are defined for removing specific punctuation marks in the `standardize_fa` function using regular
expressions.

## Text Vectorization

Text vectorization layers for English and Farsi languages are defined. `en_vectorization` for English language
uses `standardize_en` for text preprocessing and has a maximum vocabulary size of 5000 tokens. `fa_vectorization` for
Farsi language uses `standardize_fa` for text preprocessing and has a maximum vocabulary size of 5000 tokens.

## Data Tokenizing

Tokenization of text data is performed using text vectorization layers `en_vectorization` and `fa_vectorization`. The
layers are adapted to the target and input data, and the top 10 words in the vocabulary for each layer are printed. The
Farsi text data `example_input_batch` is tokenized using `fa_vectorization` layer, and the English text
data `example_target_batch` is tokenized using `en_vectorization` layer.

## Vocabulary Size

The size of the vocabulary for `fa_vectorization` is returned.

## Text to Tokens Conversion

The tokenized sequence from the `en_vectorization` and `fa_vectorization` layers is converted back to the original text
by mapping each token to the corresponding word in the vocabulary. The reconstructed text is printed.

## Model Hyperparameters

The embedding dimension and the number of units for the LSTM layer are defined.

## Model Creation: Encoder Layer

The `Encoder` class is defined, which takes `input_vocab_size`, `embedding_dim`, and `enc_units` as arguments.
The `call` method of the `Encoder` class receives a sequence of tokens and an optional initial state, passes the tokens
through the embedding layer to obtain a sequence of vectors, and passes the sequence of vectors through the GRU layer to
obtain a new sequence of vectors and an internal state. Finally, it returns the new sequence of vectors and the internal
state.

## Encoder Layer Example

An instance of the `Encoder` class is created with the Farsi text vectorization layer `fa_vectorization` as input. The
Encoder layer is applied to the tokenized Farsi text data `example_tokens` and the output and the state are saved. The
shapes of the input batch, input batch tokens, encoder output, and encoder state are printed.

## Attention Layer

The `BahdanauAttention` class is defined, which takes `units` as input. The `call` method of the `BahdanauAttention`
class receives a query tensor, a value tensor, and a mask tensor, passes the query tensor through the W1 dense layer and
the value tensor through the W2 dense layer to obtain the transformed query and value vectors, computes the attention
scores between the query vector and each value vector, applies the mask to the attention scores, and computes the
context vector as the weighted sum of the value vectors.

## Attention Layer Example

An instance of the `BahdanauAttention` class is created with `units` set to 10. The attention layer is applied to a
query tensor and a value tensor, and the output is saved. The shape of the output tensor is printed.

## Model Creation: Decoder Layer

The `Decoder` class is defined, which takes `target_vocab_size`, `embedding_dim`, `dec_units`, and `att_units` as
arguments. The `call` method of the `Decoder` class receives a sequence of tokens, an encoder output, an optional
initial state, and an optional mask, passes the tokens through the embedding layer to obtain a sequence of vectors,
passes the sequence of vectors and the initial state through the GRU layer to obtain a new sequence of vectors and a new
state, passes the new sequence of vectors and the encoder output through the attention layer to obtain a context vector,
concatenates the context vector with each vector in the new sequence of vectors to obtain a new sequence of vectors with
context, and passes the new sequence of vectors through a dense layer to obtain the logits for each token in the target
vocabulary. Finally, it returns the logits.

## Decoder Layer Example

An instance of the `Decoder` class is created with the English text vectorization layer `en_vectorization` as input, and
the hyperparameters defined earlier. The Decoder layer is applied to the tokenized English text
data `example_target_tokens`, the encoder output, and the encoder state, and the output is saved. The shape of the
output tensor is printed.

## Full Model

The `Translator` class is defined, which takes `encoder`, `decoder`, `fa_vectorization`, and `en_vectorization` as
arguments. The `call` method of the `Translator` class receives a sequence of tokens in Farsi, passes the tokens through
the encoder to obtain the encoder output and the encoder state, initializes the decoder state with the encoder state,
generates a sequence of tokens in English by iteratively applying the decoder to the previous token and the current
decoder state until the `[END]` token is generated or the maximum sequence length is reached, and finally returns the
generated sequence of tokens in English.

## Full Model Example

An instance of the `Translator` class is created with the `Encoder` and `Decoder` layers created earlier, and
the `fa_vectorization` and `en_vectorization` vectorization layers. The `translate` method of the `Translator` class is
used to translate the example Farsi text `example_input` to English, and the translated text `example_output` is
printed.

Note that the hyperparameters, text preprocessing functions, and vectorization layers can be adjusted to optimize the
performance of the model for specific use cases.
