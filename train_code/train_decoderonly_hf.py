import numpy as np
import json
import torch
from config import parse_args
from pytorch_lightning import seed_everything

from data_helper import SafetyDatasetDecoderOnly
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def sft_collator(instances):
    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    
    
if __name__ == "__main__":
    
    args = parse_args()
    
    output_dir = args.savedmodel_path
    train_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation
    learning_rate = args.learning_rate
    eval_batch_size = args.val_batch_size
    eval_steps = args.eval_step
    save_steps = args.save_step
    num_train_epochs = args.max_epochs
    warmup_steps = args.warmup_steps
    ds_config = args.ds_config
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    with open(args.train_path, 'r', encoding='utf8') as f:
        train_data = json.load(f)
    
    train_dataset = SafetyDatasetDecoderOnly(args, train_data)

    print(train_dataset[0])
    input_text = tokenizer.decode(train_dataset[0]['input_ids'].tolist())
    labels = tokenizer.decode([id for id in train_dataset[0]['labels'].tolist() if id >= 0])
    print(f'input: {input_text}\n\nlabels: {labels}')
    
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, use_cache=True, trust_remote_code=True)
    

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend='auto',
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="no",
        eval_accumulation_steps=1,
        # eval_steps=eval_steps,
        save_strategy='epoch',
        # save_strategy='steps',
        # save_steps=save_steps,
        save_only_model=True,
        report_to='tensorboard',
        load_best_model_at_end=False,
        logging_steps=20,
        deepspeed=ds_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
