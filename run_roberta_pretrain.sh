python3 run_language_modeling.py \
  --output_dir=roberta_pretrain_output \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --do_train \
  --train_data_file=$TRAIN_FILE \
  --eval_all_checkpoints \
  --do_eval \
  --eval_data_file=$TEST_FILE \
  --mlm \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --save_steps 2000 \