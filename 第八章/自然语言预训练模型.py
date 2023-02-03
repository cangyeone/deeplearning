import torch, torchtext
from torchtext.functional import to_tensor
# torchtext提供了自然语言预训练的模型 
xlmr_base = torchtext.models.XLMR_BASE_ENCODER
model = xlmr_base.get_model()# 获取模型
transform = xlmr_base.transform()# 文本数据转换
input_text = ["Hello world. How are you!", "The RoBERTa model was pretrained on the reunion of five datasets."]# 输入文本
input_id = transform(input_text)# 转换为ID 
print(input_id)
model_input = to_tensor(input_id, padding_value=1)
output = model(model_input)
print(output.shape)
# 可以使用output的向量进行文本分类或进行其他工作