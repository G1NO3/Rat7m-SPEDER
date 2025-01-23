python main.py --alg spedersac --env rat7m --feature_dim 128 \
                --max_timesteps 1000000 --dir dim128_sa_sp_buffer_20body_normalized_lasso1e-2_xyz_group\
                --eval_freq 500 --discount 0.9 --batch_size 256 --lasso_coef 0.01 --feature_lr 0.0001

# python visualize.py --alg spedersac --env rat7m --feature_dim 64 \
#                 --max_timesteps 1000000 --dir dim64_sa_sp_buffer_20body_normalzied_lasso1e-2_xyz_group --start_timesteps 20000\
#                 --eval_freq 500 --discount 0.9 --batch_size 128 --times 100
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