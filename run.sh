python main.py --alg spedersac --env kms --feature_dim 128 \
                --max_timesteps 1000000 --dir S_f128_lasso_001_dataset200_continuous_retrain_u001 \
                --eval_freq 1000 --discount 0.9 --batch_size 128 --lasso_coef 0.01 --feature_lr 0.0001
# Caution
# python visualize.py --alg spedersac --env kms --feature_dim 128 \
#                 --max_timesteps 1000000 --dir S_f128_lasso_0_dataset200_continuous_retrain --start_timesteps 200\
#                 --eval_freq 5000 --discount 0.9 --batch_size 5 --times 100 --device cpu --scale_factor 200
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