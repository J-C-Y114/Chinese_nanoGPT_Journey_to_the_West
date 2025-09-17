
# Chinese_nanoGPT_Journey_to_the_West

## 1. 介绍
训练一个开源的nanoGPT模型，参数量为12.35M，语料采用中文《西游记》原著全文，获得的文档共71.5万字。最终能输出10句以上的推理内容。

## 2. 引用说明
本项目基于[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)进行修改，调整了数据集、训练环境配置。

## 3. nanoGPT介绍
![nanoGPT](assets/nanogpt.jpg)

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## 4. 依赖安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
本项目使用课题组服务器进行训练，使用一块NVIDIA TITAN V显卡，CUDA版本为11.1.0，因此安装以下版本的依赖：
- `python 3.9.23`
- [pytorch](https://pytorch.org)
  - `torch 1.7.1+cu110`
  - `torchvision 0.8.2+cu110`
  - `torchaudio 0.7.2`
- [numpy](https://numpy.org/install/) 1.21.6
-  `transformers 4.28.0` for huggingface transformers (to load GPT-2 checkpoints)
-  `datasets 2.12.0` for huggingface datasets (if you want to download + preprocess OpenWebText)
-  `tiktoken 0.5.1` for OpenAI's fast BPE code
-  `wandb 0.15.12` for optional logging
-  `tqdm 4.66.1` for progress bars

## 5. 运行

### 数据预处理
本步骤的文件在 `data/jtw` 下。输入文本数据为 `input.txt`，采用 `UTF-8` 编码。运行以下命令：
```sh
python data/jtw/prepare.py
```
将在`data/jtw`下生成 `train.bin` 和 `val.bin`，将文本数据转换为二进制训练数据。并且会对生成数据进行统计：
```log
length of dataset in characters: 725,458
vocab size: 4,493
train has 652,912 tokens
val has 72,546 tokens
```

### 使用GPU进行训练

老版本的`Pytorch`不支持编译，因此需要关闭编译选项。采用`config/train_jtw.py`的配置进行训练。
```sh
python train.py config/train_jtw.py --compile=False
```
根据配置文件`config/train_jtw.py`，训练的上下文为480字，有384个特征值通道。采用6层Transformer，每层有6个头。一共迭代5000次。训练共耗时27分钟，最大占用显存为11.47G。最终的损失函数值为0.7076。

训练好的模型存放在 `out-jtw` 目录下。之后使用 `sample.py` 进行推理。

### 推理生成

```sh
python sample.py --out_dir=out-jtw --prompt="花果山"
```
提示词可在 `--prompt=""` 参数中修改。默认将生成10个例子，其中一个如下：

花果山中洞，山中有千里，有二十里，二百里路，下有二千余下。那马不知二十里远近，路口里不见，只见马，吆喝喝，把个头大，喊声喝道：“饶你性命！孤拐了我也！”那妖精见，口里相迎。他两个各不怕，败阵而走。

　　这行者按得空，轮着铁棒，在山崖上，呼的一声，喝道：“泼怪物！”那怪骂道：“你这泼物原来不识生在此，你这快快拿住我！”又行者与他二十余，举着，劈头就走，往东山压住。那山坡下，那妖精抵住，那山坡前乱筑。这大圣抵住八百精一个，赶上前，举着那妖精劈脸便刺。那小妖，跑将一路去，就打，往东下一下。行者正走。行者见了，急走。他两个在山坡前，又见他急急转回头就走，慌得那妖精魔打，急转身就赶，打。八戒赶他，也不住走。行者，一个孙大个在山坡下，见一阵而赶。行者引路，举着钉钯就打。那妖魔不住，将身一下，喝声“变！”那妖怪抖抖了刀乱筑，劈面，被他下一下，把八戒扳剪尾巴，慌得那妖精扎了，慌慌得行者绑上大惊道：
  
　“沙僧，你赶。”行者道：“你也不来！这厮不快走！”行者道：“有多凶了你了那里走罢！”原来那里晓得，被行者拖一齐挡住道：“你是那小妖也不怕！想是走了。”好大圣，轮着铁棒，好风岭，那里有五百里远，不见，就喊声，轮拳撞。行者劈脸打。八戒喝道：“赶我！”那怪不得，仗着的揪，抵着唐僧，举铁棒就打。那怪见行者急掣棒劈面，拨转身去，就打。那半空。行者急败阵，拨转云，赶赶来。慌得那怪的一个爪来，到洞口喷。众妖一拥而走。才按云头，见八戒、沙僧一齐呐喊，见行者对八戒道：“你这猴子，你师父已是个猴怪，二打不出洞！”那黄婆们不起，又使钉钯抵住颈项躲，一声叫道：“我！来与你跑走！”那怪道：“你去你去！我是个不字，又有谁？”行者在洞前，一齐声叫道：“大王，将你这样无名！”行者道：“我晓得我赶吾师父，教我送你一发，饶你师父！”那小妖从空，一齐跪下磕头道：“我把你在你儿索子拿你！”那怪依言，放行者方才起，跪下，双手，叫道：“行者！他是个名字？”那怪整衣物厉声大怒道：“大王！你是我师父子，吾你？”行者道：“我奉谢！恕罪！恕罪！”行者道：“你是我的大话也不知，我就敢领你罪，等只因你哩！”那怪闻言，即传旨，传旨，叫：“小钻风，都来。”那妖执兵器，那柄芭蕉扇子，一柄芭蕉扇，一柄芭蕉扇上打，落来，着牛魔一条咬，上前，骂道：“你这猴子！我问你这猴头儿！我说，名也不得！我称是弼马温！”行者道：“我称我的神王，敢称了，因姓名字，称起何来见

