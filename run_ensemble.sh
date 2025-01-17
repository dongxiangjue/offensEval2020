export MODEL_DIR=/home/ubuntu/models
export ENSEMBLE_DIR=/home/ubuntu/ensemble_dataset
export TASK_NAME=SST-2


python3 run_ensemble.py \
  --model_name_or_path $MODEL_DIR \
  --task_name $TASK_NAME \
  --bert_model_name_or_path bert-base-uncased \
  --roberta_model_name_or_path roberta-base \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir $ENSEMBLE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --save_steps 5000 \
  --overwrite_output_dir \
  --output_dir ensemble_finetune_output
