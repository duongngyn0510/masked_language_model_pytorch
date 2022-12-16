# Experiments Masked language model with Pytorch

Using `bert-base-uncased` tokenizer

For given input sequence, assign a 15% probability of each token being masked `[MASK]`. Don't mask:
+ `PAD` token having ids 0
+ `SOS` token having ids 101
+ `EOS` token having ids 102

`[MASK]` token having ids 103

This token2ids is the convention of `bert-base-uncased` tokenizer