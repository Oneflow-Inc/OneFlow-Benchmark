## Integration Test
GPT参数众多，在调试过程中经常需要进行各种测试，为了方便和规范测试，特建立此文件夹保存测试相关脚本。

测试执行的方式是：运行测试脚本，选择某一项单测或者执行所有测试用例。所有脚本和数据需要存储在集群不同机器的相同目录下，或者使用NFS，或者在容器中映射相同的目录。

下面按照文件功能进行分别介绍。

### test_util.py
定义了各种测试所依赖公共函数。

#### `choose_and_run_test_cases`
选择并执行测试用例，这是所有测试需要调用的唯一接口函数，需要如下输入：
- filepath：指定的配置文件地址，配置文件是一个csv文件，每一行代表一个测试用例，每一列代表这个测试所需要设置的参数。
- handler：是一个函数对象，`choose_and_run_test_cases`会将用户选定的某个测试用例的相关参数（也就是csv文件的某一行）作为handler的输入，handler根据这些参数生成测试能用的参数，然后再调用`gpt_run`运行测试。
- port：机器之间通信时可能需要指定的端口号。
- nsys=''：如果需要的话可以指定`nsys`命令，这段命令将被加在master将要被执行的命令前面，目的就是为了生成`qdrep`文件。

#### `gpt_run`
根据`handler`生成的参数信息运行测试，如果是多机，先通过ssh把测试命令通过ssh发送到从机启动测试（nohup运行），然后启动主机的运行脚本（非nohup）。

### 性能测试
- 脚本文件：`performance_test.py`
- 配置文件：`config_of_performance_test.csv`
- 运行目录：`$BENCHMARK_ROOT/LanguageModeling/gpt-2`
运行方式：
```
python3 ./integration_test/performance_test.py
```

或指定配置文件目录
```
python3 ./integration_test/performance_test.py --cfg /path/to/config/file.csv
```

运行后屏幕输出：
```
0: n1g1_1x1x1
1: n1g8_8x1x1
2: n1g8_4x2x1
3: n1g8_2x4x1
4: n1g8_1x8x1
5: n2g8_2x8x1
6: n4g8_4x8x1
7: n1g1_1x1x1
8: n1g8_8x1x1
9: n1g8_4x2x1
10: n1g8_2x4x1
11: n1g8_1x8x1
12: n2g8_2x8x1
13: n4g8_4x8x1
14: n2g4_2x4x1
Please Choose a number, 'a' for all, otherwise quit: 
```

然后通过键盘选择输入`0`到`14`数字运行某一个测试用例，或者输入`a`运行所有测试用例，否则会直接退出。这里的编号顺序和csv配置文件相同。

### 参数介绍
- `--cfg`: 指定csv配置文件路径
- `--python_cmd`: 指定使用的`python`命令位置，如某个`/root/anaconda3/envs/python36/bin/python`
- `--port`: 指定多机ssh的端口号。多机容器环境启动ssh时如果限定了端口就需要配置该参数。
- `--workspace`: 指定工作目录。主机的工作目录是`LanguageModeling/GPT`，多机时要求所有从机有相同的工作目录，通过此参数设置。缺省支持容器环境下的`/workspace`目录。
