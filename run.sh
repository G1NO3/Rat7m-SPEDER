# python main.py --alg hilp --env kms --feature_dim 64 \
#                  --max_timesteps 1000000 --dir A_f64_task78_lr1e-4_hilp \
#                  --eval_freq 1000 --discount 0.9 --batch_size 16 --lasso_coef 0.01 --feature_lr 0.0001 \
#                  --actor_type gaussian
# Caution
python visualize.py --alg opal --env kms --feature_dim 64 \
               --max_timesteps 1000000 --dir A_f64_task78_lr1e-3_opal_shallowopal --start_timesteps 200\
               --eval_freq 5000 --discount 0.9 --batch_size 5 --times 100 --device cuda:0 --scale_factor 200\
               --actor_type autoregressive
# for i in 64 96 128
# do  
    # echo "Running with feature dim: $i"
    # python main.py --alg spedersac --env rat7m --feature_dim $i \
    #             --max_timesteps 1000000 --dir dim${i}_sa_sp_buffer_20body_normalized\
    #             --eval_freq 1000 --discount 0.9 --batch_size 256
    # python visualize.py --alg spedersac --env rat7m --feature_dim $i \
    #           --max_timesteps 1000000 --dir dim${i}_sa_sp_buffer_20body_normalized\
    #           --eval_freq 1000 --discount 0.9 --batch_size 256
# done
