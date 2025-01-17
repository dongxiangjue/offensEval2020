python3 run_glue.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --save_steps 10000 \
  --output_dir roberta_fine_tune_output

