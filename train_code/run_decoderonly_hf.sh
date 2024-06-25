data_dir=./sft_data/combine_rules_qwen_format_label_first
model_name=Qwen1.5-14B-Chat
model_name=Qwen-14B-Chat
model_dir=/data/zhangzhexin/huggingface_pretrained_models/${model_name}
max_epoch=3
deepspeed --include localhost:2,3,5,6 --master_port=20954 \
    train_decoderonly_hf.py --ds_config=ds_config_hf.json \
    --train_path=${data_dir}/train.json \
    --model_dir=${model_dir} \
    --pretrained_model_path= \
    --batch_size=12 --val_batch_size=12 \
    --gradient_accumulation=1 \
    --savedmodel_path=./hf_save/${data_dir}/seed12_${model_name}_bs48_warm0_lineardecay_lr2e-5_maxlen1536_max${max_epoch} --ckpt_file='' \
    --max_epochs=${max_epoch} --warmup_steps=0 --warmup_ratio=0 \
    --learning_rate=2e-5 --fp16= \
    --seed=12 \
    --max_length=1536 --eval_step=75 --save_step=100 \
    --lr_decay=linear --patience=1 \
    --ema=0 --ema_start_epoch=0
