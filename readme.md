# 多mask版本

正常使用(num_mask == 1)和之前兼容，仅列举num_mask > 2的改动。

##### 生成值

Logits：一个batch中，模型预测的mask_pos位置对应的lm_logit，shape为`batch_size, num_mask, vocab_size`

-   最后一个batch可能不满
-   如果sentence_fn中没有<mask>，则默认使用的是不基于MLM的学习方法，这个返回值将为None

Hidden_states: 一个batch中，每一层的所有hidden states，shape为`batch_size,  num_layer + 1, seq_len, hidden_size`

Mask_pos： 一个batch中，每一句句子的mask位置，shape为`batch_size, num_mask`。与Logits相同，如无mask token则返回None



## More details...
For more technical details, check out our [paper](https://arxiv.org/abs/2201.03514)

If you meet any problem, you can seek our help in the Wechat group.
