# python main.py --alg spedersac --env kms --feature_dim 1 \
#                 --max_timesteps 1000000 --dir SPEDER_pure_feature_ctrl_f1_nonoise_mildnormalized_divideby200\
#                 --eval_freq 1000 --discount 0.9 --batch_size 512 --lasso_coef 0.01 --feature_lr 0.0001
# Caution
python visualize.py --alg spedersac --env kms --feature_dim 512 \
                --max_timesteps 1000000 --dir SPEDER_pure_feature_ctrl_f512_nonoise_mildnormalized_divideby200 --start_timesteps 200\
                --eval_freq 5000 --discount 0.9 --batch_size 5 --times 100 --device cpu --scale_factor 200
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