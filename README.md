# MSAIC
Codes for Microsoft AI Challenge 2018

## Problem statement
Given a question and 10 passages related with only one of them having the answer to the question,
and find the passage containing that answer.

### Evaluation
The evaluation metric for this is [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)

## BM25
This is a [Best Match 25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm.
It uses tf-idf values to rank the passages and then find the passage with highest rank.

