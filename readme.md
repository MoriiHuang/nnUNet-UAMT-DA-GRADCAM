### nnUNet with semi-supervised method(UA_MT),distance oriented attention model and grad-cam

authored by Huang Chuanyi (SJTU439)

### introduction

This is an extension of [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master), implementing the UA-MT, distance-oriented attention mechanism model and the grad-cam migration to nnUNet

### installation

git clone or download this rep,then
```shell
cd nnUNet
pip install -e .
pip install grad-cam
```

### Data Struct
```
nnUNet_raw
    - imagesTr(训练集)
    - imagesTs(测试集)
    - labelsTr(训练集label)
    - imagesUns(未标注数据，半监督)
```
### Running Script
```shell
### 数据预处理
nnUNetv2_plan_and_preprocess -d idx  --verify_dataset_integrity  -unsupervised True
### baseline
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold
### attention
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold -tr nnUNetTrainer -attention True
### UA-MT
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold -tr nnUNetTrainerUAMT -unsupervised True
### Grad-CAM
nnUNetv2_predict -i input folder -d 1 -c 3d_fullres -o output folder -cam True
```
