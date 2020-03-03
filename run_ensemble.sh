export BERT_CONFIG_DIR=/home/ubuntu/bert_config
export ROBERTA_CONFIG_DIR=/home/ubuntu/roberta_config

python3 run_ensemble.py \
  --model_name_or_path $MODEL_DIR \
  --task_name $TASK_NAME \
  --bert_config_name $BERT_CONFIG_DIR \
  --roberta_config_name $ROBERTA_CONFIG_DIR \
  --bert_tokenizer_name $BERT_CONFIG_DIR \
  --roberta_tokenizer_name $ROBERTA_CONFIG_DIR \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir $ENSEMBLE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --save_steps 5000 \
  --output_dir ensemble_finetune_output
