from transformers import BertTokenizer

#加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

####################ckip

# from transformers import (
#   BertTokenizerFast,
#   AutoModel,
# )
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
# model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')

########################

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

#print("tokenizer : ",tokenizer)
#print("sents : ",sents)

#编码两个句子
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,#不指定返回數據類型，默認是list
)

#print("out : ",out)

#print("tokenizer.decode : ",tokenizer.decode(out))

#增强的编码函数
out = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,
    add_special_tokens=True,

    #可取值tf,pt,np,默认为返回list(tensorflow,pytorch,numpy)
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
# for k, v in out.items():
#     print(k, ':', v)

tokenizer.decode(out['input_ids'])

#批量编码句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=15,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
# for k, v in out.items():
#     print(k, ':', v)

tokenizer.decode(out['input_ids'][0]), tokenizer.decode(out['input_ids'][1])

#批量编码成对的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
# for k, v in out.items():
#     print(k, ':', v)

tokenizer.decode(out['input_ids'][0])

#获取字典
zidian = tokenizer.get_vocab()

print(type(zidian), len(zidian), '月光' in zidian)

#添加新词
tokenizer.add_tokens(new_tokens=['月光', '希望'])

#添加新符号
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

zidian = tokenizer.get_vocab()

#print(type(zidian), len(zidian), zidian['月光'], zidian['[EOS]'])

#编码新添加的词
out = tokenizer.encode(
    text='月光的新希望[EOS]',
    text_pair=None,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=8,
    return_tensors=None,
)

#print(out)

#print(tokenizer.decode(out))

#from datasets import load_dataset

#dataset = load_dataset(path='seamew/ChnSentiCorp')

#print(dataset)

#保存数据集到磁盘
#注意：运行这段代码要确保【加载数据】运行是正常的，否则直接运行【从磁盘加载数据】即可
#dataset.save_to_disk(dataset_dict_path='./data/ChnSentiCorp')

#从磁盘加载数据
from datasets import load_from_disk

dataset = load_from_disk('./data/ChnSentiCorp')

print(dataset)

#取出训练集
dataset = dataset['train']

#print(dataset)

print(dataset[0])

#sort

#未排序的label是乱序的
print(dataset['label'][:10])

#排序之后label有序了
sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])

#shuffle

#打乱顺序
shuffled_dataset = sorted_dataset.shuffle(seed=42)

print(shuffled_dataset['label'][:10])

#select
print(dataset.select([0, 10, 20, 30, 40, 50]))

#filter
def f(data):
    return data['text'].startswith('选择')


start_with_ar = dataset.filter(f)

print(len(start_with_ar), start_with_ar['text'])

#train_test_split, 切分训练集和测试集
dataset.train_test_split(test_size=0.1)

#shard
#把数据切分到4个桶中,均匀分配
dataset.shard(num_shards=4, index=0)

#rename_column
dataset.rename_column('text', 'textA')

#remove_columns
dataset.remove_columns(['text'])

#map
def f(data):
    data['text'] = 'My sentence: ' + data['text']
    return data


datatset_map = dataset.map(f)

datatset_map['text'][:5]

#set_format
dataset.set_format(type='torch', columns=['label'])

print(dataset[0])

from datasets import load_dataset
#第三章/导出为csv格式
dataset = load_dataset(path='seamew/ChnSentiCorp', split='train')
dataset.to_csv(path_or_buf='./data/ChnSentiCorp.csv')

#加载csv格式数据
csv_dataset = load_dataset(path='csv',
                           data_files='./data/ChnSentiCorp.csv',
                           split='train')
csv_dataset[20]

#第三章/导出为json格式
dataset = load_dataset(path='seamew/ChnSentiCorp', split='train')
dataset.to_json(path_or_buf='./data/ChnSentiCorp.json')

#加载json格式数据
json_dataset = load_dataset(path='json',
                            data_files='./data/ChnSentiCorp.json',
                            split='train')
json_dataset[20]