#!/env/bin/activate
export CUDA_VISIBLE_DEVICES=0

model_name=NeuralBasis

learning_rate=0.0001
batch_size=128
e_layers=4
d_model=64

seq_len=96
down_sampling_layers=3
down_sampling_window=2
d_ff=32

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
