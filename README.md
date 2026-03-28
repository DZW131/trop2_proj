# trop2_proj
上海复旦肿瘤细胞核/膜分割 | 基于SAM2模型

训练脚本：python training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_trop2_me_nu.yaml --num-gpus 1

推理脚本：
对单张图片输出结果：python infer.py --img_path assets/2025_10_30_10_55_56_511990_103761_29646.png --save_res --model bplus_menu
对trop2/test测试集计算指标：python infer.py --mode test --eval --model trop2_me_nu --save_res
