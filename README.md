<h1 align="center">TorchFCOPE</h1>

TorchFCOPE是一个以几乎零代码改动的FCPE用于预测开口度的纯粹玩具性的实验仓库。
！！！注意：本仓库为可行性测试仓库，不会进行维护，代码不做任何保证！！！

## 预处理

0.**数据获取**

在预处理之前，你应该获得数据，使用[LipsSync](https://github.com/KCKT0112/LipsSync)获取数据，文件结构应该类似

```
2025-02-04_22-01-52/
    audio.wav
    mouth_data
```

1.**重采样/更改文件结构**

将以上文件夹复制到`op/raw`文件夹下，如

```
op/raw/
    2025-02-04_22-01-52/
        audio.wav
        mouth_data.csv
    2025-02-04_22-43-56/
        audio.wav
        mouth_data.csv
```

运行`python resample_audio.py --config train/configs/config.yaml`，这会对音频重采样并且修改文件名，之后的文件结构如下

```
op/train/1/
    audio/
        2025-02-04_22-01-52.wav
        2025-02-04_22-43-56.wav
    csv/
        2025-02-04_22-01-52.csv
        2025-02-04_22-43-56.csv
```

2.**预处理**

运行`python preprocess.py --config train/configs/config.yaml`

`fbl_onnx_path`为FoxBreatheLabeler权重的路径，你可以在https://github.com/openvpi/dataset-tools/releases/tag/FblModel 获取该模型

`mask_breath`为对换气声部分mask的方式，true为对换气声mask，false为对换气声反mask

`ope_mode`为对ope取值的三种方式（1.'jawOpen' 2.'jawOpen - mouthClose' 3.'jawOpen * (1 - mouthClose)'）

3.**对预处理结果切片**

运行`python fixop.py`

## 训练

```
python train/train_wav.py  --config train/configs/config.yaml
```