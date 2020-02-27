python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 96 \
  --per_gpu_eval_batch_size 96 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --save_steps 2000 \
  --output_dir bert_finetune_output
