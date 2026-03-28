# trop2_proj
上海复旦肿瘤细胞核/膜分割 | 基于SAM2模型

训练脚本：python training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_trop2_me_nu.yaml --num-gpus 1

推理脚本：
对单张图片输出结果：python infer.py --img_path assets/2025_10_30_10_55_56_511990_103761_29646.png --save_res --model bplus_menu
对trop2/test测试集计算指标：python infer.py --mode test --eval --model bplus_menu --save_res

权重文件：
权重存放在Hugging Face的该仓库中
https://huggingface.co/DZW666/trop2_weights/tree/main
<img width="1091" height="361" alt="image" src="https://github.com/user-attachments/assets/b35c75de-1339-4c77-a411-8259fc392274" />
其中
https://huggingface.co/DZW666/trop2_weights/blob/main/sam2.1_hiera_base_plus.pt
是SAM2官方预训练B+权重，重新训练时要放到checkpoints/文件夹下
https://huggingface.co/DZW666/trop2_weights/blob/main/1118_me_nu2_checkpoint.pt
是该项目训练得到的权重，使用时要注意其名字和路径要与训练/测试中脚本的名字路径一样。
