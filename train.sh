nohup python train.py  --name DBTC \
    --lr 0.0002  --scale_factor 8 --load_size 128 \
    --dataroot /data1/CelebA/celeba_train   --batch_size 1 --total_epochs 100 \
    --print_freq 10 --save_latest_freq 500 &
# name 实验的名字 不重要，随便改
# lr 学习率 learning rate
# scale_factor 超分倍数 scale
# load_size 输入大小 input size
# dataroot 训练集路径 training data root
# batch_size batch大小
# total_epochs 训练总epoch数
# print_freq 每几次迭代print一次
# save_latest_freq 每几次迭代保存一个ckpt

# 多卡训练时在train.py第7行指定多卡 如[0,1,2,3]