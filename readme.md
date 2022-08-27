# Baseline

## Prepare your environment

The implementation of Black-Box Tuning is quite simple, you can check our code and easily implement it in your own environment. Or you can create a new environment to run our implementation, which is based on `pycma`, `Transformers` and `FastNLP`. Optionally, you can use `fitlog` to monitor experimental results. You can uncomment the fitlog-related lines in our code to use it.

```bash
conda create --name bbt python=3.8
conda activate bbt
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
pip install cryptography
```

## Using BBT

Now you can run Black-Box Tuning with the following code:

```bash
python bbt.py --seed 42 --task_name 'sst2'
```

Then the best prompt will be saved to `./results`.
Run `test.py` to get the predictions on the test set.

## Test script

### 输入接口

```python
def test_api(
    sentence_fn: Callable[[str], str], 
    test_data_path: str,
    embedding_and_attention_mask_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]=None, 
    hidden_states_and_attention_mask_fn:
    	Callable[[int, Tensor, Tensor], Tuple[Tensor, Tensor]]=None,
    task_name: str='SST-2', 
    batch_size: int=32, 
    device: Optional[str, torch.device]='cuda': 
) -> Tuple[Union[Tensor, None], Tensor, Union[Tensor, None]]:
    ...
```

==这个函数是一个生成器==



##### Sentence_fn

对test set中的句子添加（自然语言组成的）离散模板的函数，以待预测的句子作为参数，返回处理后的句子。见下例：

```python
pre_str = tokenizer.decode(list(range(1000, 1050))) + ' . '
middle_str = ' ? <mask> , '

def sentence_fn(test_data):
    return pre_str + test_data + middle_str + test_data
```

-   注：sentence_fn的参数并非真实的数据，只是一个占位符，因此不能直接对数据的长度做操作。在测试阶段，我们统一把所有长于256 token的句子切割到256个token。



##### test_data_path

赛方提供的加密测试数据路径



##### embedding_and_attention_mask_fn

**适用于仅在embedding层之后加入的soft prompt**，通过操作embedding layer得到的表征和对应的attention_mask并按相同顺序返回，其出现在RobertaEmbedding和RobertaEncoder层之间。见下例：

```python
 def embedding_and_attention_mask_fn(embedding, attention_mask):
        prepad = torch.zeros(size=(1, 1024), device=device)
        pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        return embedding + torch.cat([prepad, best_prompt, pospad]), attention_mask
```

-   Shape: embedding `bsz, seq_len, hidden_size`, attention_mask `bsz, seq_len`



##### hidden_states_and_attention_mask_fn

**适用于在embedding层之后和除最后一层encoder layer得到的结果中均加入的soft prompt**（每层的prompt内容可以不同）见下例：

```python
 def hidden_states_and_attention_mask_fn(i, embedding, attention_mask):
        prepad = torch.zeros(size=(1, 1024), device=device)
        pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        return embedding + torch.cat([prepad, best_prompt[i], pospad]), attention_mask
```

-   i: 是在`[0, num_layer)`中的整数，代表prompt出现的层数，其中i=0对应embedding层之后
-   Shape: embedding `bsz, seq_len, hidden_size`, attention_mask `bsz, seq_len`

该函数在一次forward中被调用`num_layer`次



==以上两个函数可以均为None，或其中一个为None，对应的函数将不起作用，但两参数都非None是意义不明的行为，因此会报错==



##### task_name

['SNLI', 'SST-2', 'MRPC', 'AGNews', 'Yelp', 'TREC']中的一个，需要与测试文件路径匹配



##### batch_size

测试采用的batch_size



##### device

测试时使用的模型和中间生成Tensor的device



##### 生成值

Logits：一个batch中，模型预测的mask_pos位置对应的lm_logit，shape为`batch_size, vocab_size`

-   最后一个batch可能不满
-   如果sentence_fn中没有<mask>，则默认使用的是不基于MLM的学习方法，这个返回值将为None

Hidden_states: 一个batch中，每一层的所有hidden states，shape为`batch_size,  num_layer + 1, seq_len, hidden_size`

Mask_pos： 一个batch中，每一句句子的mask位置，与Logits相同，如无mask token则返回None



## More details...
For more technical details, check out our [paper](https://arxiv.org/abs/2201.03514)

If you meet any problem, you can seek our help in the Wechat group.
