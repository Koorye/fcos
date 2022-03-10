# fcos

基于pytorch的FCOS复现，实现对PASCAL VOC数据集的训练和预测

## 使用方法

启动Visdom
```shell
python -m visdom.server
```
开始训练
```shell
python train.py
```

训练时可访问[Visdom](http://localhost:8097)查看训练进度
![](examples/%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E5%8F%AF%E8%A7%86%E5%8C%96.png)
