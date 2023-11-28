# PosFL-DL

## Introduction

​	目前，在现实的大场景任务中，联邦学习和蒸馏学习结合是一个有前景的应用。经典的蒸馏学习做法是通过logits传递模糊的知识（如果能够结合一些创意方法则可以解决Non-IID问题），这可以弥补联邦学习中模型参数大带来的占用显存问题。但同时会有例如单点故障的问题或者是存在恶意节点的攻击，区块链是目前比较火热的解决方式，它通过其共识机制，加密算法对整个过程起到了保护的作用。
    

​	本方法即采用POS机制区块链对蒸馏学习提供的logits或者是通过自蒸馏的MEME模型（通过模型ACC，余弦相似度，grad距离等因素）进行有权重的reward奖励和聚合策略，这将是一个有效的激励机制同时能够保证解决下游任务的需求。并尝试提供一个在可靠POS区块链基础之上的联邦学习平台

## Contribution

    1.创新的联邦蒸馏学习方法(互相蒸馏) -> 能够解决NonIID问题
    2.新的reward机制（参考FAIR-BFL） -> 能够更加有效的解决安全
    https://www.semanticscholar.org/reader/c365d11185883f311c6937f270185d2faa6433d3
    3.

## Code

```
main.py - 入口文件
BlockChain.py - 提供区块链的结构
Block.py - 提供区块的结构
Device.py - 设置基础3种角色的共同属性（主要包括加密解密的部分-RSA...）
Worker.py - 继承Device.py 本地训练两个模型或者一个模型
Val.py - 继承Device,py 做验证的功能，主要是验证worker节点给过来数据的合理性和安全性
Miner.py - 继承Device.py 做挖矿功能，整合所有内容，
ChainCode.py - 智能合约的功能，自动发放奖励，通过代币数量选取miner
model.py - 存放基础模型
README.md - 解释文件
env_cuda_latest.yaml - 需要安装的文件内容
example_sh.sh - shell脚本后台运行代码，跑baseline
```
其中不使用print，使用log来打印日志
## Step

1. 完成FedAvg，并给出可靠的激励机制。（解决恶意节点问题，增加分配的合理性，可以在其他维度例如模型某一层参数或者是logits进行分析就可以给出合理的度量手段）DataSharply？
2. 使用蒸馏学习的方法（解决模型参数大问题）