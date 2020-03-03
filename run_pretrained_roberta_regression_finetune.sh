export MODEL_PATH=/home/ubuntu/checkpoint-0
export TASK_NAME=OFFENSE-R
export GLUE_DIR_F=/home/ubuntu/pt_ft_r_dataset
python3 run_glue.py \
  --model_type roberta \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir $GLUE_DIR_F/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 82 \
  --per_gpu_eval_batch_size 82 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --save_steps 5000 \
  --overwrite_output_dir \
  --output_dir pretrained_roberta_regression_finetune_output