# GPT模型转换

### PyTorch模型转OneFlow模型
  - `meta.proto`，是为生成模型目录下的`meta`文件，需要执行`protoc --python_out=. meta.proto`后生成`meta_pb2.py`，即可`import meta_pb2 as meta_pb`
  ```
    syntax = "proto2";
    package gpt;

    message Shape {
        repeated int32 dim = 1;
    }

    enum DataType {
        kInvalidDataType = 0;
        kChar = 1;
        kFloat = 2;
        kDouble = 3;
        kInt8 = 4;
        kInt32 = 5;
        kInt64 = 6;
        kUInt8 = 7;
        kOFRecord = 8;
        kFloat16 = 9;
        kTensorBuffer = 10;
    }

    message Meta {
        required Shape shape = 1;
        required DataType data_type = 2 [default = kFloat16];
    }
  ```
  - 转换脚本`convert_pt_to_of_gpt.py`，执行`python3 convert_pt_to_of_gpt.py --py_model_dir /path/to/iter_0500000/mp_rank_00/model_optim_rng.pt`即可在当前目录下的`convert_pt_to_of_gpt`生成OneFlow模型
    - `--py_model_dir`,pytorch模型地址
    - `--of_dump_path`,保存转换后的模型路径
  
  