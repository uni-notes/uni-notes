# Text Classification

- Devise features by hand: Does the message contain “church”. Does the email contain an Indian organization’s domain
- Bag of words: Count of occurrences off each word of a pre-defined ‘vocabulary’

Pre-Processing

- Stemming: only keep the root of the word
  - “slowly” and “slow” both mapped to “slow”
- Filtering
  - Stopwords: articles
  - Filler words
  - rare words



$$
\begin{aligned}
\text{tf(term)} &= \dfrac{n_\text{term}}{n_\text{terms in document}} \\
\text{idf(term)} &= \ln \left \vert \dfrac{n_\text{documents}}{n_\text{documents containing term}}
\right \vert \\
\text{tf-idf(term)} &= \text{tf(term)} \times \text{idf(term)}
\end{aligned}
$$

