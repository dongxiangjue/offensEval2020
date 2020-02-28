python3 run_language_modeling.py \
  --output_dir=bert_pretrain_linebyline_final_output \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --train_data_file=$TRAIN_FILE \
  --eval_all_checkpoints \
  --do_eval \
  --eval_data_file=$TEST_FILE \
  --mlm \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --save_steps 100000 \
  --block_size 128 \
  --line_by_line \
  --overwrite_cache \
  --overwrite_output_dir \
  --num_train_epochs=10

