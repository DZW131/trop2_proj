import json
import os
from collections import namedtuple, OrderedDict

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from infer_utils import get_prompt
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferByRt(nn.Module):
    def __init__(self, rt_path):
        super().__init__()
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        # with open(rt_path, 'rb') as f:
        #     engine = runtime.deserialize_cuda_engine(f.read())
        with open(rt_path, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(
                    f.read(4), byteorder="little"
                )  # read metadata length
                metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
            except UnicodeDecodeError:
                f.seek(0)  # engine file may lack embedded Ultralytics metadata
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        input_names = []
        fp16 = False  # default updated below
        dynamic = False
        is_trt10 = not hasattr(model, "num_bindings")
        num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
        for i in num:
            if is_trt10:
                name = model.get_tensor_name(i)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    # self.input_name = name
                    input_names.append(name)
                    if -1 in tuple(model.get_tensor_shape(name)):
                        dynamic = True
                        context.set_input_shape(
                            name, tuple(model.get_tensor_profile_shape(name, 0)[2])
                        )
                        if dtype == np.float16:
                            fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_tensor_shape(name))
            else:  # TensorRT < 10.0
                name = model.get_binding_name(i)
                self.input_name = name
                dtype = trt.nptype(model.get_binding_dtype(i))
                is_input = model.binding_is_input(i)
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(
                            i, tuple(model.get_profile_shape(0, i)[1])
                        )
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        print("precision:", "FP16" if fp16 else "FP32")
        # batch_size = bindings[self.input_name].shape[0]  #if dynamic,this is instead max batch size
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, input):
        # YOLOv5 inference
        # b, ch, h, w = im.shape  # batch, channel, height, width


        if (
            self.dynamic
            and input[self.input_names[-1]].shape != self.bindings[self.input_names[-1]].shape
        ):
            for input_name in self.input_names:
                self.context.set_input_shape(input_name, input[input_name].shape)
                self.bindings[input_name] = self.bindings[input_name]._replace(
                    shape=input[input_name].shape
                )
            for output_name in self.output_names:
                self.bindings[output_name].data.resize_(
                    tuple(self.context.get_tensor_shape(output_name))
                )
        # s = self.bindings[self.input_name].shape
        # assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        for input_name in self.input_names:
            self.binding_addrs[input_name] = int(input[input_name].data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = {x: self.bindings[x].data for x in sorted(self.output_names)}
        if isinstance(y, (list, tuple, dict)):
            return (
                self.from_numpy(y[0])
                if len(y) == 1
                else {x: self.from_numpy(y[x]) for x in y}
            )
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):

        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


if __name__ == "__main__":
    onnx_path = "./sam2_mul_output_fp32.engine"
    img_path = "assets/0000.png"
    prompts_json_path = img_path.replace(".png", ".json")
    img = Image.open(img_path)
    from torchvision.transforms import Normalize, Resize, ToTensor
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tensor = (ToTensor()(img))[None, ...]
    image = np.array(img.copy().convert("RGB"))

    data = {
        "key": [
            item["points"]
            for item in json.load(open(prompts_json_path))["shapes"]
            if item["label"] == "肿瘤细胞膜"
        ]
    }
    prompts = {}
    prompts = get_prompt(data, prompts, "key")
    ptl = np.concatenate(prompts["key"], axis=0)
    point_coords = torch.tensor(ptl[:, None, :2]).cuda().to(torch.float32)
    point_labels = torch.tensor(ptl[:, 2:]).cuda().to(torch.int32)
    # model = InferByRt(onnx_path)
    encoder_model = InferByRt("sam2.encoder_fp32.engine")
    decoder_model = InferByRt("sam2.decoder_fp32.engine")
    
    # with open(img_path, 'rb') as f:
    #     value_buf = f.read()
    # img = imfrombytes(value_buf, float32=False)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # img = data_process(img)
    # img = transforms.functional.to_tensor(img).unsqueeze(0).cuda()

    # a = torch.from_numpy(img).to(device).repeat(1,1,1,1).data_ptr()
    import time

    from torch.nn import functional as F

    input = {
        "image": img_tensor.cuda(),
        "point_coords": point_coords,
        "point_labels": point_labels,
    }
    
    output = encoder_model(input)
    input.update(output)
    start = time.time()
    times = 1
    for i in range(times):
        output = decoder_model(input)

    end = time.time()
    print(f"time: {(end - start) / (times) * 1000} ms")
    print(output["low_res_masks"].shape)
    for i in range(output["low_res_masks"].shape[1]):
        masks = output["low_res_masks"][:, [i], :, :]
        masks = (
            F.interpolate(
                masks.to(torch.float32).cpu(),
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
            )
            > 0
        )
        mask = masks.sum(0)[0].cpu().numpy()
        mask = mask * (255 // mask.max())
        cv2.imwrite(f"a_{i}.jpg", mask.astype(np.uint8))
