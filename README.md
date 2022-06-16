# H-M-Personalized-Fashion-Recommendations
rank: 116/2952 —— the Program of this competition
# HM 商品推荐复盘

比赛网址：https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

#### 比赛介绍

以以往的数据以及顾客和商品的元数据来开发产品推荐，例如服装类型和客户年龄，到产品描述的文本数据，再到服装图片的图像数据，

特征提取：

+ 多模态，文本以及图像
+ 表格数据
+ 最终的方案就是，表格数据抽取特征为主，多模态，文本图形为辅助

评价标准（MAP@12）：预测一个人再一周以后的购买情况。也就是推出top12

#### 数据说明

图像

item：10w中服装

用户数量：

用户和items的交互数据：3000w条数据

#### 方案策略

+ csv保存为pickle，可以节省时间
+ 特征工程是本次比赛的重点：
  + 创建user-items矩阵，使用LightFM库，训练了user的embedding
  + 对商品的属性特则会那个做ont-hot编码，并入交易表，然后groupby user聚合。构成了user和商品属性之间的关系特征。
  + 通过上述新特征，生成candidates候选项，并且衍生出少量rank特征
  + 还生成了user的静态特征，user动态特征，item静态特征，item动态特征，user-item对静态特征，user-item对动态特征
+ 训练集分配


#### 比赛难点

提取有用特征，多模态还是特征工程

#### 代码

特征提取用的是：lightFM模型，loss用的是bpr loss

商品召回的主要策略：
+ repurchase
  + 10**9 * day_rank + volume_rank（我们认为时间越近越重要）
    + day_rank是购买的日期距离
    + volume_rank是购买的重复购买的数量
  + 以此排序取全部值
+ item2item
  + 和repurchase的方法一样，不一样的就是取的值
  + 每个用户取至少前12个item

+ popular
  + 选取购买最多的商品，的前60个
  + 提取全部用户
  + 每个用户对应每个商品，以此创建candidates
+ category popular
  + 选取购买数量最多的商品类别
  + 以item2item的商品，用用户-类别对的数量作为groupby，再结合商品特征作为candidates

最后，从 candidates_category_popular 中 drop 那些在 candidates_repurchase 出现过的 user-item 对


特征提取包括：
+ 静态特征
  + 年龄
  + idx结尾特征
+ 动态特征
  + 交易表中，每个用户在某个时间段的[price 和 sales_channel_id]的[平均值,标准差]
  + 交易表中，每个item在某个时间段的[price 和 sales_channel_id]的[平均值,标准差]
  + 交易表某段week，加入年龄列
  + item新鲜度特征
  + 每个item被购买量
  + user新鲜度特征
  + 每个user购买量
  + user-item对新鲜度特征
  + user-item购买量
  + lfm features

模型主要用的是：
+ lightBGM
+ CatBoost

