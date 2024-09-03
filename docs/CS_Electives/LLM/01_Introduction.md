# Introduction

## Text Generation Methodologies

- $n$-grams
  - Bigrams
  - Trigrams
- Bag of words
- Bag of tokens; token = subwords
	- Byte pair encoding

Tokenization causes issues
- LLM cannot spell words
- LLM cannot perform simple string processing tasks, such as reversing a string
- LLM performs worse in non-English languages
- LLM is bad at simple arithmetic
- LLM prefers YAML over JSON with LLMs
- LLM breaks due to special/unstable tokens
	- `<|endoftext|>`
	- trailing whitespace
	- `SolidGoldMagikarp`
	- special tokens
- LLM is not end-to-end language modelling

## Architectures

- MLP
- RNN
- GRU
- LSTM
- Transformers