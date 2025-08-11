for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
  for bs in 8 16 32; do
    for accum in 4 8 16; do
      rm openwebtext_transformer_ckpt.pt
      python train.py --base_lr $lr --batch_size $bs --accumulation_steps $accum --num_steps 200 --curve_path "curve_lr${lr}_bs${bs}_accum${accum}.npy"
    done
  done
done