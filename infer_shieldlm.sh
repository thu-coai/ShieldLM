model_path=thu-coai/ShieldLM-14B-qwen
model_base=qwen # baichuan, internlm, chatglm
# model_path=ShieldLM-13B-baichuan2
# model_base=baichuan
# model_path=ShieldLM-7B-internlm2
# model_base=internlm
# model_path=ShieldLM-6B-chatglm3
# model_base=chatglm
lang=en # or zh

# input as file
CUDA_VISIBLE_DEVICES=0 python infer_shieldlm.py \
    --model_path ${model_path} \
    --input_path examples/example_input.jsonl \
    --output_path examples/example_output.jsonl \
    --rule_path examples/example_rules.txt \
    --model_base ${model_base} \
    --batch_size 4 \
    --lang ${lang}

# input as single query and response
CUDA_VISIBLE_DEVICES=0 python infer_shieldlm.py \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --lang ${lang}