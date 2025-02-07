# python main.py --alg spedersac --env kms --feature_dim 64 \
#                 --max_timesteps 1000000 --dir dim64_lasso1e-2_trainfromscratch\
#                 --eval_freq 1000 --discount 0.9 --batch_size 256 --lasso_coef 0.01 --feature_lr 0.0001

python visualize.py --alg spedersac --env kms --feature_dim 64 \
                --max_timesteps 1000000 --dir dim64_lasso1e-2_trainfromscratch --start_timesteps 20000\
                --eval_freq 5000 --discount 0.9 --batch_size 128 --times 100 --device cpu
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