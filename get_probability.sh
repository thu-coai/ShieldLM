model_path=thu-coai/ShieldLM-14B-qwen
model_base=qwen # baichuan, internlm, chatglm
# model_path=thu-coai/ShieldLM-13B-baichuan2
# model_base=baichuan
# model_path=thu-coai/ShieldLM-7B-internlm2
# model_base=internlm
# model_path=thu-coai/ShieldLM-6B-chatglm3
# model_base=chatglm
lang=en # or zh

CUDA_VISIBLE_DEVICES=0 python get_probability.py \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --lang ${lang} # or zh