import torchtext 
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer 
# 构建子词分割模型模型
torchtext.data.functional.generate_sp_model(
    "data/en.news.txt", 
    vocab_size=20000, 
    model_type='unigram', 
    model_prefix='ckpt/spmodel')

# 加载子词分割模型
spmodel = load_sp_model("ckpt/spmodel.model")


# 定义文本
texts = ["This architecture takes an image as input and resizes it to 448*448 by keeping the aspect ratio same and performing padding.", 
      "This architecture uses Leaky ReLU as its activation function in whole architecture except the last layer where it uses linear activation function."] 

# 子词进行分割
gen_id = sentencepiece_tokenizer(spmodel)  
text_id = gen_id(texts)
print(list(text_id))

# 子词分割并转换为子词ID
gen_id = sentencepiece_numericalizer(spmodel)  
text_id = gen_id(texts)
print(list(text_id))

