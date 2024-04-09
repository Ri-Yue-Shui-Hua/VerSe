
### Training WorkFlow
1. Preprocess the data
```c
preprocess/reorient_image.py
```
从数据链接下载数据，通过预处理代码，将数据和标签处理成指定的格式。

## 1、下载数据集

数据集按文件夹包括样本

| 文件夹 | 样本数 |
| --- | --- |
| train | 80 |
| validate | 40 |
| test | 40 |

### 1.1、VerSe 2019 (subject based data structure)

下载脊柱分割数据集

## Data Link
Public dataset:
- (VerSe'19 train): https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip
- (VerSe'19 validation): https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip
- (VerSe'19 test): https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip
