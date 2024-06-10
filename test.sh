
python test.py --gpus 1 --name DBTC --dataset_name single \
    --load_size 128 --dataroot /data1/New_Helen/Bicubic   \
    --pretrain_model_path ./best.pt \
    --save_as_dir ./result_helen

# name 实验的名字 不重要，随便改
# dataset_name 指定data文件夹下具体使用哪个文件预处理测试集，不需要改（single对应data/single_dataset.py)
# load_size 输入大小 input size
# dataroot 测试集的位置 testing data root
# pretrain_model_path 预训练模型的位置
# save_as_dir 存储测试图片的位置