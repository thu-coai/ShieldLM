import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--valid_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--genout_path', type=str, default='')
    parser.add_argument('--ds_config', type=str, default='')

    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=64, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=2, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=8, type=int, help="num_workers for dataloaders")
    
    # ======================== Load Pretrained =========================
    parser.add_argument('--load_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default='')

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str)
    parser.add_argument('--ckpt_file', type=str, help='save prefix for ckpt file')
    parser.add_argument('--ckpt_path', type=str, help='fine-tuned model path')

    
    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs')
    parser.add_argument('--instruct_type', type=int, default=0, help='How many epochs')

    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--warmup_ratio', default=0, type=float, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--train_all_data', type=bool, default=False, help='train all data')
    
    parser.add_argument("--lr_decay", default='linear', type=str, help="Weight deay if we apply some.")
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--patience", default=1, type=float, help="Early Stop.")
    parser.add_argument("--gradient_accumulation", default=1, type=int, help="Early Stop.")
    parser.add_argument("--eval_step", default=100, type=int, help="Early Stop.")
    parser.add_argument("--save_step", default=100, type=int, help="Early Stop.")


    parser.add_argument('--model_dir', type=str, default='')
    # parser.add_argument('--max_input_length', type=int, default=256)
    # parser.add_argument('--max_output_length', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=256)
    # parser.add_argument('--max_output_length', type=int, default=128)
    
    parser.add_argument('--fp16', type=bool, default=False, help='use fp16')
    parser.add_argument('--ema', type=float, default=0)
    parser.add_argument('--ema_start_epoch', type=int, default=0)

    
    return parser.parse_args()
