---
title: 目标检测与智慧制造 笔记（1）
date: 2022-12-13
author: Kevin 吴嘉文
category:
- 知识笔记
tag:
- CV
---

## PPDET 笔记

落地效果：具体查看 PaddleDetection 仓库

 **模块回顾：** 

DCN，SPP，FPN，PAN（PA），Coordconv，Mish，EMA 指数滑动平均，DropBlock，batch norm，dark，Heatmap，REID，

 **其他笔记：** 

+ 使用预训练模型时，对应的 lr 可以调低点，如 0.1 倍等。

### 移动端

mobilene V3 

小模型需要注意召回率的提升

+ 剪裁 + 蒸馏，精度基本无损或有提升

 **PP-PicoDET** 

超小型的目标检测模型，可以用于移动端部署，他被应用在了 PP-Tinypose 等应用场景上。支持 320 像素检测，在骁龙 865 上可以达到 150 FPS；640 像素可以达到 20 FPS

算法优化很多，如 NAS 或 EA 算法对网络结构进行子网搜索，优化整体网络大小。

###  **关键点检测**  

 [AI studio link - 人体关键点分析 coco 数据集](https://aistudio.baidu.com/bj-cpu-01/user/902220/5195641/notebooks/5195641.ipynb)

 **关键点检测任务：** 

+ 关键点位置检测：一般采用 17 个点，点位置可以自定义
+ 关键点链接（结构）信息判断；
+ 关键点检测更多的可以与规则进行配合，完成检测任务。

 **难点问题：** 

- 真实数据无法采集或者采集困难，标注量大，种类多等问题。可以考虑关键点检测。

- 姿势判断：通过肢体形态，建立规则来判断姿态。动作判断：是一个时序问题。

 **两种方案：** 

+  **Top-Down：** 先检测人体框，而后检测各个框对应的关键点。HRNet（High Resulution Net），DarkPose 等
+  **Bottom-Up：** 检测所有点，而后将点分配到各个人体上。SWAHRNet, HrHRNet

 **数据结构（以 coco 人体关键点检测为例[link](https://aistudio.baidu.com/aistudio/datasetdetail/9663))：** 

+ 大小：144406 个样本，每个样本包含一张 crop 图片和 17 个关键点，每张训练集的图片都是经过目标检测裁剪的，因此一张原始输入图片通常能够生成 3-4 个 crop 图片。

 **模型结构：** 

+ backbone 编码
+ 使用 Dark 热力度回归（当然也可以对关键点的 xy 坐标直接进行回归。）
  + 设置好超参数 $\sigma$，以关键点为中心，生成服从高斯分布的热力图。
  + 现阶段为了平衡内存和量化误差，一般热图大小是输入大小的 1/4
  + 如果一张 input 有 17 个关键点，那么 target 将会是 17 张热力图
  + 模型直接预测热力图，训练时候优化 mse，解码时候 argmax 提取关键点 。

 **移动端模型**   **PP-Tinypose**  （122 FPS，支持 6 人以上同时检测，可部署在移动端上）

+ Top-Down，算力要求相对较少。
+ PP-PicoDet -> AID + Dark Encoding + UPD unbias -> 骨干网络 ShuffleNet -> 小尺寸输入 + 高精度解码 -> 工程优化

### 多目标跟踪

 **参考代码：** PaddleDetection 中的 PP-Tracking；[AI studio](https://aistudio.baidu.com/aistudio/projectdetail/2096117?channelType=0&channel=0)

数据需求：

参考 [基于 FairMOT 实现人流量统计](https://aistudio.baidu.com/aistudio/projectdetail/2421822?channelType=0&channel=0) 案例；共 53694 张已标注好的数据集用于训练

 **任务形式：** 

+ 在连续帧中，维持被检测物个体的 ID 信息，并记录其轨迹

+ 目标检测 -> ReID -> 相似度计算 + 数据关联 -> 匈牙利匹配，卡尔曼滤波 ->输出轨迹

 **单镜头 MOT 两种代表算法：** 

+  **DeepSORT：** 检测后 REID；分离式算法 SDE（DeepSORT）落地方案成熟，鲁棒性高，相对 JDE 速度慢。
+  **JDE， FairMOT：** 检测+REID；实时跟踪系统通常采用联合式 JDE（FairMOT 算法)，速度快，对数据集敏感，落地未普及

 **SDE DEEPsort 算法核心：** 

+ 目标检测，预测当前帧里的物体；
+ 进入 REID 阶段，截取目标图片，计算其对应的 embedding 特征，作为匈牙利算法距离计算方式之一。

+ 利用匈牙利匹配，在 cost 距离最小的情况下，为当前帧的所有检测物体匹配上前一帧的所有检测物体。
  + 物体的距离计算方式：通过位置状态、reid 表征特征和 IOU 匹配等数据，联合计算两个检测物体之间各个数值的距离（cosine 或马氏距离等，可自定义）。
  + 根据 deepsort 算法的匹配规则进行检测物体匹配：如连续 3 帧轨迹匹配后，才会进行确认匹配。当确认的轨迹和检测框连续失配 30 次后，才会被删除

+ 卡尔曼滤波

  + 能够在噪声存在的情况下，对系统的下一步动向做一个大胆的预测。

  + 资源消耗小

### 旋转框检测 TODO

### 跨境头跟踪 TODO

对同一场景下，不同摄像头拍摄的视频，进行多目标跟踪	

+ 目前可以实现的效果如何？

数据难点：

+ 数量受限：标注数据少；
+ 质量受限，遮挡，分辨率底等；摄像头多样，角度多样；

 **方案：** 

目标检测：

+ 采用各种优化手段，使用高性能模型

ReID：

+ 采用 PP-LCNET，去除最后的 FC 层，而后将 feature 映射到 512 维度上。之后使用 AUGMIX, ARCMARGIN, SupConLoss 等方法来训练，提高分类效果。

轨迹融合

+ 依赖大量先验知识，减小搜索空间，提升匹配效率。
  + 摄像头之间的相对关系
  + 目标不会同时出现在不同镜头下
  + 目标运动符合基本规则
+ 轨迹过滤（不动的轨迹，不可能存在轨迹的区域）
+ 轨迹相似度计算（reRank）
+ 轨迹聚类（相同类别物体轨迹聚类）

### 目标检测案例

 **污染物残留、刮伤检测** 

+ 2600+ 缺陷图片，缺陷种类 10+，缺陷种类样本数量从 56-1000 张不等

小批量快速验证：

1. 先标注小部分的样本数据，使用基础模型尝试预测效果。 **若 mAP 达到 80%+ 则说明深度学习可以用来处理该问题。** 
2. 标注全部数据，进行模型训练优化与调整，进行 **数据增广方案** 
3. 额外的数据增广方案：对缺陷进行变形、缩放后，复制到其他的图片上，增加数据集。

 **大尺幅线阵相机下的缺陷检测** 

线阵相机常见尺度较大，像素较高，有时候一张图片就有 100+M。该场景下，需要同时考虑到图像的预测速度，精度以及对大尺寸图像的预处理方法。

图片处理：

+ 大尺寸输入需要对图片进行裁剪：如何裁剪？
  + 可以通过识别每张完整图片需要的时间确定。如裁剪成 4 张的话，每次推理的时间限制就会是裁剪成 16 张的 4 倍。
  + 使用模型的像素限制是多少？`512*512`, `768*768`？推理设备的限制是多少？
  + 可以通过[模型选型工具](https://www.paddlepaddle.org.cn/smrt)来筛选：
    + 根据要求 input-size 大小，target-size 大小，推理速度来选择模型。

### 行人分析 PP-Human

参考 PaddleDetection 下的 PP-Human 代码；该部分自由创作的空间非常大。

 **部分应用以及技术** 

| 技术分类                 | 应用                                         |
| ------------------------ | -------------------------------------------- |
| 目标检测 + 分类属性分析  | 危险区域人员闯入，人员特征识别               |
| 跟踪 + 分类属性分析      | 简单动作分析（可通过图片判断的），人流量统计 |
| 跟踪 + 关键点 + 行为分析 | 复杂动作分析（仅能通过动作判断）             |
| 跟踪 + 视频分类          |                                              |
| 跟踪+ReID（多镜头）      | 酒店可疑人员行踪分析                         |
|                          |                                              |

 **检测+属性分析：** 

+ 通过 det 截取人物图像；对人物图像进行多标签分类，如有 7 个属性，则进行 7 次二分类（属性>threshold 即可）。

 **跟踪 + 属性分析：** 

+ 通过 mot 模型实现目标跟踪；
+ 截取目标图像，进行属性多标签分类；（同上）；
+ 对截取目标图像进行额外的图像分类或目标检测，来判断是否吸烟等。
  + paddlehuman 中使用了 额外的 detection 来判断是否吸烟。（检测到相关物品，就判断存在某种行为）
  + 同属性检测，通过画面的某一帧来判断某种行为是否存在。

 **跟踪+关键点+行为分析：** 

+ 通过 mot 模型实现目标跟踪；
+ 截取目标图像，进行属性多标签分类；（同上）；
+ 通过 hrnet 对 crop input 进行关键点提取。
+ 通过 关键点信息，传入 STGCN 进行动作分析。

 **跟踪+ReID（多镜头）：** 

+ 

 **问题难点** 

+ 真实业务场景中，工厂训练数据少。

 **解决方案之一：** 基于关键点信息，进行图像分类任务 [案例链接](https://aistudio.baidu.com/aistudio/projectdetail/4061642)

+ 进行人体关键点检测。
+ 截取关键点对应部位的图像，如头部关键点，则截取的图像能够覆盖头部。
+ 用截取的图片检测对应的内容，如用头部照片检测工人是否佩戴安全帽、口罩等
  + 带帽子的视为头部正样本，没带的视为头部负样本

部署在 nvidia jetson 设备上，tensorrt7.0, ubuntu 18, cuda10.2, jetson4.5



[YOLO 2 aistudio](https://aistudio.baidu.com/aistudio/projectdetail/1922155?channelType=0&channel=0)

paddlepaddle detection 的整体文件加架构于 paddleocr 相似，主文件夹在 ppdet 下，文件加中分别提供了各种模型的 architectures，CV 常用的 losses，Components（neck，backbones，data augmentaion）等

仓库入口：`./tools/train.py`

### 模型选择

[paddle 产业选型工具](https://www.paddlepaddle.org.cn/smrt)

+ 输入：场景任务、需求指标（预测时间，精度和速度权衡，部署系统，硬件输入）、标注文件、数据特点（均衡性、大小、目标长款比文件等）
+ 输出：模型推荐、数据分析、硬件推荐

轻量网络：

+ det：picodet
+ cls：pplcnet
+ 关键点检测：pp-tinypose

### 数据标注

####  **EISeg 自动标注** 

工业质检中，尝尝需要知道缺陷的面积，因此语义分割任务尝尝被用于工业质检中。

语义分割交互式分割标注方案（EISeg）：

+ 类似 PS 的魔术棒工具，软件会尝试自动切割工具。
+  **软件的智能检测是基于铝制材质缺陷设计的。** 

#### labelme

paddle 支持 labelme 格式转换为各种格式。

### 训练方法

#### 配置 config

config 中定义了：

+ 模型预训练权重地址
+ 模型架构，如 backbone，neck，head 以及特殊层的维度大小
+ 输入维度，transform 方法，data aug 方法，batch size 等
+ 优化器超参，如 epoch，learning rate，scheduler，optimizer，正则方案等）

根据显卡以及 batch size 来调整对应的超参配置。`__base__` optimizer 中

同时调整训练数据格式。`config/dataset` 文件加中



#### 开始训练

```bash
python tools/train.py -c configs/ppyolo/ppyolo.yml --eval
```

#### 验证

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo.yml -o weights=output/ppyolo/best_model
```

#### 推理模型

```bash
# 推理单张图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理目录下所有图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_dir=demo
```

Python 中采用 `os.system` 执行推理

```bash
from tqdm import tqdm
# 这里是使用单卡的示例代码
!CUDA_VISIBLE_DEVICES=0
# !python tools/infer.py -c configs/ppyolo/ppyolo_tiny_650e_coco_roadsign.yml \
# -o weights=output/ppyolo_tiny_650e_coco_roads_best_model/best_model.pdparams \
# --infer_img=/home/aistudio/data/images/road301.png

for img in tqdm(infer_imgs):
    os.system("python tools/infer.py -c configs/ppyolo/ppyolo_tiny_650e_coco_roadsign.yml \
        -o weights=output/ppyolo_tiny_650e_coco_roads_best_model/best_model.pdparams \
        --infer_img=/home/aistudio/data/images/" + img)
```



#### 其他

可选：在训练之前使用`tools/anchor_cluster.py`得到适用于你的数据集的 anchor，并修改`configs/ppyolo/ppyolo.yml`中的 anchor 设置



## 备注

目标检测数据格式：

VOC 格式下

```bash
├── Annotations/ # xml 格式标注
├── ImageSets/
│   └── Main/
│       ├── train.txt # 训练集名字,在训练时不太会用到
│       └── valid.txt 
├── JPEGImages/ # 原图
├── train.txt   # 原图-标注
└── valid.txt
└── label_list.txt
```



```xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000051.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>291539949</flickrid>
	</source>
	<owner>
		<flickrid>kristian_svensson</flickrid>
		<name>Kristian Svensson</name>
	</owner>
	<size>
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>motorbike</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>352</xmin>
			<ymin>138</ymin>
			<xmax>500</xmax>
			<ymax>375</ymax>
		</bndbox>
	</object>
	<object>
		<name>motorbike</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>105</xmin>
			<ymin>1</ymin>
			<xmax>427</xmax>
			<ymax>245</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>1</difficult>
		<bndbox>
			<xmin>415</xmin>
			<ymin>61</ymin>
			<xmax>465</xmax>
			<ymax>195</ymax>
		</bndbox>
	</object>
</annotation>
```


