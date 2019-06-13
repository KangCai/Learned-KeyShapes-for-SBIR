# Learned-KeyShapes-for-SBIR

[English](https://github.com/KangCai/Learned-KeyShapes-for-SBIR/blob/master/README.md)

很高兴你发现了这个项目，该项目是对一个论文算法实现的工程，原论文是 "Sketch based Image Retrieval using Learned KeyShapes (LKS)"。欢迎提出任何意见或建议。

---

### 算法介绍

总体流程如下图所示，

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/1.png"/>

图最上面一排，从 Sketch dataset 到 Sketch Patch Extraction 再到 Keyshape Generation 得到 keyshapes，属于步骤 1，即关键图形模板的生成，
该部分使用的数据集是一个特定的、与应用场景数据无关的简笔画数据集；图下面两排，到KeyShape Detection这一步为止，是根据步骤 1 得到的关键图形模板，
分别对图像和简笔画提取关键图形的过程，属于步骤 2；最后是 LKS-Histogram Computation，即在步骤 2 的基础上计算 LKS 直方图描述子。

**1. 关键图形生成**

这一步重点是提取 DAISY 特征，然后 K 聚类，

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/2.png"/>

**2. 分别对图像和简笔画提取关键图形**

这一步是根据步骤 1 得到的关键图形模块来提取，值得一提的是简笔画可以直接提取，但对于图像，需要先对原始图像提取 sketch token image，然后进一步
设定阈值提取 contour image，如下所示，

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/3.png"/>

提取关键图形的方法是，对 stroke points 提取 DAISY 描述子，然后统计与该描述子距离最近的关键图形，表示形式如下所示，

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/4.png"/>

**3. 在步骤 2 的基础上计算 LKS 直方图描述子**

分三小步，第一小步是按相似度距离给对应的关键图形投票，距离度量采用的是高斯核距离，如下

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/5.png"/>

，投票方式就是简单地对相应的关键图形进行桶投票，如下

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/6.png"/>

；第二小步是将图像划分成网格来进行处理，这一小步跟大多数特征提取方式一样，主要是考虑图像的空间结构；第三小步是将上述特征进行单位归一化，
相似性度量方法不是采用的通常使用的 Minkowski Distance（闵可夫斯基距离，包含一组距离度量方式），包括L1距离（曼哈顿距离）、
L2距离（欧式距离)、切比雪夫距离，而是 Hellinger distance，为了的是消除某些桶上峰值的影响，运算过程是先取的平方根，再算 L2 距离，如下

<img src="https://raw.githubusercontent.com/KangCai/Learned-KeyShapes-for-SBIR/master/images/doc/7.png"/>

---
