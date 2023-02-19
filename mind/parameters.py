import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    # ddp
    parser.add_argument("--enable_ddp", type=utils.str2bool, default=True)
    parser.add_argument("--ddp_master_port", type=str, default="12355")
    parser.add_argument("--world_size", type=int, default=8)

    parser.add_argument("--mode", type=str)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)

    # trainer
    parser.add_argument("--trainer", type=str, default="GPTTrainer")
    
    parser.add_argument("--news_file", type=str)
    parser.add_argument("--news_emb_file", type=str)
    parser.add_argument("--test_news_file", type=str, default=None)
    parser.add_argument("--test_news_emb_file", type=str, default=None)

    parser.add_argument("--train_behavior_file", type=str)
    parser.add_argument("--eval_behavior_file", type=str)
    parser.add_argument("--test_behavior_file", type=str)
    
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--cache_dir", type=str, default='./cache')

    parser.add_argument("--eval_first", type=utils.str2bool, default=True)
    parser.add_argument("--test_news_batch_size", type=int, default=512)
    parser.add_argument("--test_user_batch_size", type=int, default=512)
    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--filter_num", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=100)

    # model training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2022)
    

    parser.add_argument(
        "--num_words_title", type=int, default=32
    )
    parser.add_argument(
        "--user_log_length", type=int, default=50,
    )
    parser.add_argument(
        "--news_dim", type=int, default=400,
    )
    parser.add_argument(
        "--gpt_news_dim", type=int, default=1536,
    )
    parser.add_argument(
        "--news_query_vector_dim", type=int, default=200,
    )
    parser.add_argument(
        "--user_query_vector_dim", type=int, default=200,
    )
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="drop rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="number of steps to save checkpoints")

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test",
    )
    # pretrain
    parser.add_argument(
        "--use_pretrain_news_encoder", type=utils.str2bool, default=False
    )
    parser.add_argument("--pretrain_news_encoder_path", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="FastFormer")

    nrms_parser = parser.add_argument_group("nrms")
    nrms_parser.add_argument("--num_attention_heads", default=20, type=int)

    plm_parser = parser.add_argument_group('plm')
    plm_parser.add_argument("--model_type", default="bert", type=str)
    plm_parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)
    plm_parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name. ",
    )
    plm_parser.add_argument(
        "--config_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    plm_parser.add_argument(
        "--tokenizer_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    plm_parser.add_argument("--bert_output_layer", type=int, default=12)
    plm_parser.add_argument(
        "--bert_trainable_layer", 
        type=int, nargs='+',
        default=[8, 9, 10, 11],
        choices=list(range(12)))
    plm_parser.add_argument(
        "--copy_model_path",
        default=None,
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    plm_parser.add_argument(
        "--copy_model_type",
        default='bert',
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
