yellow_str = lambda x: f"\033[93m{x}\033[0m"
# model = build_sam2("configs/sam2/sam2_hiera_t.yaml")
def count_trainable_params(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    trainable_params_info = {
        "model_sz": f"{round(all_params/1e9, 3)}B",
        "all_params": f"{round(all_params/1e6, 1)}M",
        "trainable_params": f"{round(trainable_params/1e6, 1)}M",
        "trainable_params_ratio": f"{100 * trainable_params / all_params:.2f}%",
    }
    for k,v in trainable_params_info.items():
        print(yellow_str(f"{k}: {v}"))
    return trainable_params_info