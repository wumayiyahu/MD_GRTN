## 1. 目标
对齐MD-GRTN代码与论文的超参数配置，确保模型实现符合论文描述。

## 2. 实施
- [ ] 1.1 配置文件： `configurations/PEMS04_astgcn.conf`，将`num_layers`从2修改为3（STFormer层数）。
- [ ] 1.2 训练代码： `train_md_grtn.py`，在损失函数部分添加Huber Loss实现和配置支持。
- [ ] 1.3 主训练代码： `train_md_grtn.py`，更新损失函数选择逻辑，支持`huber_loss`选项。
