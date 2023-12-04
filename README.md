# PosFL-DL

## Introduction
​	本方法即采用POS机制区块链对蒸馏学习提供的logits或者是通过自蒸馏的MEME模型（通过模型ACC，余弦相似度，grad距离等因素）进行有权重的reward奖励和聚合策略，这将是一个有效的激励机制同时能够保证解决下游任务的需求。并尝试提供一个在可靠POS区块链基础之上的联邦学习平台

## Contribution

    1.创新的联邦蒸馏学习方法(互相蒸馏) -> 能够解决NonIID问题

    2.新的reward机制（参考FAIR-BFL） -> 能够更加有效的解决安全
    https://www.semanticscholar.org/reader/c365d11185883f311c6937f270185d2faa6433d3

## Code

```
dataset
    ANY dataset
        - rawdata
        - test (.npz)
        - train(.npz)
        - config.json
utils
    -data_utils.py
- main.py 入口文件
- BlockChain.py 提供区块链的结构
- Block.py 提供区块的结构
- Device.py 设置基础3种角色的共同属性（主要包括加密解密的部分-RSA...）
- Worker.py 继承Device.py 本地训练两个模型或者一个模型
- Valdator.py 继承Device,py 做验证的功能，主要是验证worker节点给过来数据的合理性和安全性
- Miner.py 继承Device.py 做挖矿功能，整合所有内容，
- Model.py 存放基础模型
- README.md  
- env_cuda_latest.yaml 需要安装的文件内容
- example_sh.sh shell脚本后台运行代码，跑baseline
```
