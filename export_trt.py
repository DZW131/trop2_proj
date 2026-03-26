import gc
import glob
import os
import os.path as osp
import random
from typing import Any

import torch


class Exporter:
    """
    定义一个类, 专门导出各种格式的模型
    """

    def __init__(self, model, device=torch.device("cuda:0"), *args, **kwargs) -> None:
        self.model = model
        self.device = device
        self.input_info = kwargs.get("input_info", "input_info must be provided")
        self.metadata = {
            "author": "wxl",
            "description": "nnUNet export tensorrt",
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.input_tensors = {
            k: torch.rand(v).to(self.device) for k, v in self.input_info.items()
        }
        self.export_tensorrt(*args, **kwargs)

    def export_tensorrt(
        self, output_path, dynamic=True, INT8_calibrator=None, verbose=False, scale = 1
    ):
        # assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = (
            self.export_onnx()
            if not self.model.endswith(".onnx")
            else (self.model, None)
        )
        # Python API 使用tensorrt
        # The Python API can be accessed through the tensorrt module:
        import tensorrt as trt

        # 1、创建记录器 The Build Phase 要创建生成器, 必须首先创建记录器
        logger = trt.Logger(
            trt.Logger.INFO
        )  # Python绑定包括一个简单的记录器实现, 它将所有的重要消息记录到stdout  #or trt.Logger.INFO  也可以根据自己需要创建自己的记录器
        trt.init_libnvinfer_plugins(logger, "")
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE
        # 2、创建生成器
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        # 3、设置最大工作区大小
        workspace = int(20 * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
        config.default_device_type = trt.DeviceType.GPU
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        # 4、创建生成器之后, 在进行网络优化之前首先需要定义自己的网络结构
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )  # The EXPLICIT_BATCH flag is required in order to import models using the ONNX parser
        # half = builder.platform_has_fast_fp16 and self.args.half
        # int8 = builder.platform_has_fast_int8 and self.args.int8
        # TensorRT supports two modes for specifying a network:  explicit batch and implicit batch.
        # 5、Importing a Model Using the ONNX Parser  通过onnx模型 解析网络模型
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(f_onnx)  # f_onnx onnx模型路径

        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not success:
            print("failed!!!")
            pass  # Error handling code here
        # 5、创建引擎 作用是构建配置, 指定TensorRT应该如何优化模型
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        # 6、打印输入输出节点信息
        for inp in inputs:
            print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

        # To perform inference, deserialize the engine using the Runtime interface. Like the builder, the runtime requires an instance of the logger.
        if dynamic:  # 如果是动态输入, 需要创建动态输入的配置
            # scale = 150  # 动态尺寸的范围控制
            profile = builder.create_optimization_profile()
            for inp in inputs:
                shape = self.input_info[inp.name]
                profile_shape = {"min": [], "opt": [], "max": []}
                for i, sp in enumerate(inp.shape):
                    if sp == -1:
                        profile_shape["min"].append(shape[i])
                        profile_shape["opt"].append(shape[i] * scale)
                        profile_shape["max"].append(shape[i] * scale * 2)
                    else:
                        profile_shape["min"].append(sp)
                        profile_shape["opt"].append(sp)
                        profile_shape["max"].append(sp)

                profile.set_shape(
                    inp.name, **profile_shape
                )  # 最小尺寸, 常用尺寸, 最大尺寸 name, 输入节点的名字, input_names
            config.add_optimization_profile(profile)
            print("Dynamic shapes enabled")
        if INT8_calibrator:  # 如果使用INT8量化, 需要创建校准器
            from export_int8_calibrator import (
                ImageBatchStream,
                PythonEntropyCalibrator as int8_calibrator,
            )

            config.set_flag(trt.BuilderFlag.INT8)  # 设置量化类型
            # config.set_calibration_profile(profile)
            # config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            image_list = glob.glob("root/**/*.png", recursive=True)
            random.shuffle(image_list)
            image_list = image_list[:1000]
            config.int8_calibrator = int8_calibrator(
                ["input"], ImageBatchStream(4, image_list)
            )  # int8_calibrator自定义校准器
        # elif builder.platform_has_fast_fp16:# 如果平台支持FP16
        #     print("FP16 enabled")
        #     config.set_flag(trt.BuilderFlag.FP16)#设置量化类型
        # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)  # 强制使用 NHWC
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(
            output_path, "wb"
        ) as t:  # 保存engine模型
            t.write(engine if is_trt10 else engine.serialize())
            print(f"Successfully exported TensorRT engine to {output_path}")
        return output_path


if __name__ == "__main__":
    root = "."
    onnx_path = osp.join(root, "sam2.decoder.onnx")
    engine_path = onnx_path.replace(".onnx", "_fp32.engine")
    export = Exporter(
        onnx_path,
        device=torch.device("cuda:0"),
        input_info={
            # "image": (1, 3, 1024, 1024),
            "high_res_feats_0": (1, 32, 256, 256),
            "high_res_feats_1": (1, 64, 128, 128),
            "image_embed": (1, 256, 64, 64),
            "point_coords": (1, 1, 2),
            "point_labels": (1, 1),
        },
    )
    export(engine_path, dynamic=True, verbose=False, INT8_calibrator=False, scale=125)
