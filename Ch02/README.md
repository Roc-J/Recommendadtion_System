## 利用用户行为数据  

实现个性化推荐的最理想情况是用户在注册的时候就告诉我们他们喜欢什么，但是存在3个缺点  

* 目前的NLP技术很难理解用户描述的自然语言  
* 用户的兴趣是不断变化的，但用户不会随时去更新兴趣需求  
* 很多用户其实并不知道自己的兴趣所在，也不知道怎么来描述    

基于用户行为数据的应用-最典型的就是各种各样的排行榜，这些排行榜包括热门排行榜和趋势排行榜。（虽然仅仅基于简单的用户行为统计）  

用户行为数据的分析是很多优秀产生设计的基础，个性化推荐算法通过对用户行为的深度分析，可以给用户带来更好的网站使用体验。  

#### 协同过滤  
基于用户行为分析的推荐算法是个性化推荐系统的重要算法，学术界一般将这种类型的算法称为协同过滤算法。顾名思义，协同过滤就是指用户可以齐心协力，通过不断地和网站互动，使自己的推荐列表能够不断过滤掉自己不感兴趣的物品，从而越来越满足自己的需求。  

### 用户行为数据  
1. 用户行为数据在网址上最简单的存在形式就是日志。  
2. 用户行为在个性化推荐系统中一般分两种  
    * 显性反馈行为(explicit feedback)  
    显性反馈行为包括用户明确表示对物品喜好的行为。比如评分、喜好/不喜欢  
    * 隐形反馈行为(implicit feedback)  
    隐形反馈行为指的是那些不能明确反映用户喜欢的行为。最具代表性的隐形反馈行为就是页面浏览行为。用户浏览一个物品的页面并不代表用户一定喜欢这个页面展示的物品，比如可能因为这个页面链接显示在首页，用 户更容易点击它而已。  

按照反馈的明确性分，用户行为数据可以分为**显性反馈**和**隐性反馈**，但按照反馈的方向分，又可以分为**正反馈**和**负反馈**  
正反馈指用户的行为倾向于指用户喜欢该物品，而负反馈指用户的行为倾向于指用户不喜欢该物品。  
在显性反馈中，很容易区分一个用户行为是正反馈还是负反馈。而在隐性反馈行为中，就相对难以确定。  

互联网的用户行为有很多种，比如浏览网页、购买商品、评论、评分等。 

#### 用户行为的统一表示  
* user id 产生行为的用户的唯一标识  
* item id 产生行为的对象的唯一标识  
* behavior type 行为的种类（比如是购买还是浏览）   
* context 产生行为的上下文，包括时间和地点等  
* behavior weight 行为的权重（如果是观看视频的行为，那么这个权重可以是观看时长；如果是打分行为，这个权重可以是分数）  
* behavior content 行为的内容（如果是评论的内容，那么就是评论的文本，如果是打标签的行为，就是标签）  

针对不同的行为给出不同表示。  

**代表性的数据集**  

* **无上下文信息的隐性反馈数据集** 每一条行为记录仅仅包含用户ID和物品ID。Book-Crossing就是这种类型的数据集
* **无上下文信息的显性反馈数据集** 每一条记录包含用户ID、物品ID和用户对物品的评分。
* **有上下文信息的隐性反馈数据集** 每一条记录包含用户ID、物品ID和用户对物品产生行为的时间戳。Lastfm数据集就是这种类型的数据集。
* **有上下文信息的显性反馈数据集** 每一条记录包含用户ID、物品ID、用户对物品的评分和评分行为发生的时间戳。Netflix Prize提供的就是这种类型的数据集。

## 用户行为分析

**用户活跃度和物品流行度的分布**  
用户行为符合**长尾分布**  
为了说明用户行为的长尾分布，选择Delicious和CiteULike数据集一个月的原始数据进行分析。结论是不管是物品的流行度还是用户的活跃度，都近似于长尾分布。特别是物品流行度的双对数曲线，非常的接近直线。  

**用户活跃度和物品流行度的关系**  
一般来说，不活跃的用户要么是新用户，要么是只来过网站一两次的老用户。  
新用户倾向于浏览热门的物品，因为他们对网站还不熟悉，只能点击首页的热门物品，而老用户会逐渐开始浏览冷门的物品。  

仅仅基于用户行为数据设计的推荐算法一般称为协同过滤算法。学术界对协同过滤算法进行了深入研究，提出了很多方法，比如**基于邻域的方法**，**隐语义模型**，**基于图的随机游走算法**  

使用最广泛的算法是基于邻域的方法，包含

* 基于用户的协同过滤算法 这种算法给用户推荐和他兴趣相似的其他用户喜欢的物品  
* 基于物品的协同过滤算法 这种算法给用户推荐和他之前喜欢的物品相似的物品  

## 实验设计和算法评测

评测推荐系统有3种方法

* 离线实验  
* 用户调查  
* 在线实验  

### 数据集  
本章采用GroupLens提供的MovieLens数据集。  
MovieLens数据集有3个不同的版本，本章选用中等大小的数据集。
该数据集包含6000多用户对4000多部电影的100万条评分。  
该数据集是一个评分数据集，用户可以给电影评5个不同等级的分数（1-5分）。  
下面着重研究隐反馈数据集中的TopN推荐问题，因此忽略了数据集中的评分记录。也就是说，TopN推荐的任务是预测用户会不会对某部电影评分，而不是预测用户在准备对某部电影评分的前提下会给电影评多少分。  

### 实验设计  
训练集和测试集的分割  
将用户行为数据集按照均匀分布随机分成M份（本章取M=8），挑选一份作为测试集，将剩下的M-1份作为训练集。  
在训练集上建立用户兴趣模型，并在测试集上对用户行为进行预测，统计出相应的评测指标。为了保证评测指标并不是过拟合的结果，需要进行M次试验，并且每次都使用不同的测试集。然后将M次实验测出的评测指标的平均值作为最终的评测指标。  

```
import random
def SplitData(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0,M) == k:
            test.append([user, item])
        else:
            train.append([user, item])
    return train, test
```

每次实验选取不同的k（0<=k<=M-1）和相同的随机数种子seed，进行M次实验就可以得到M个不同的训练集和测试集，然后分别进行实验，用M次实验的平均值作为最后的评测指标。这样做主要是防止某次实验的结果是**过拟合**的结果（over fitting），但如果数据集够大，模型够简单，为了快速通过离线实验初步地选择算法，也可以只进行一次实验。  

### 评测指标

对用户u推荐N个物品(记为R(u)），令用户u在测试集上喜欢的物品集合为T(u)，然后可以通过准确率/召回率评测推荐算法的**精度**    
**召回率**描述有多少比例的用户-物品评分记录包含在最终的推荐列表中，而准确率描述最终的推荐列表中有多少比例是发生过的用户-物品评分记录。  

```
def Recall(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit/(all*1.0)

def Precision(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit/(all*1.0)
```

除了评测推荐算法的精度，本章还计算了算法的覆盖率，覆盖率反映了推荐算法发掘长尾的能力，覆盖率越高，说明推荐算法越能够将长尾中的物品推荐给用户。   
覆盖率就是表示最终的推荐列表中包含多大比例的物品。  

```
def Coverage(train, test, N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items)*1.0)
```

最后，我们还需要评测推荐的新颖度，这里用推荐列表中物品的平均流行度推荐结果的新颖度。如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。  

```
import math
def Popularity(train, test, N):
    item_popularity = dict()
    for user, items in train.items():
        if item not in item_popularity:
            item_popularity[item] = 0
        item_popularity[item] += 1
    
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            ret += math.log(1+item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret

```

在计算平均流行度时对每个物品的流行度取对数，这是因为物品的流行度分布满足长尾分布，在取对数后，流行度的平均值更加稳定。  

## 基于邻域的算法  

基于邻域的算法是推荐系统中最基本的算法。  
基于邻域的算法分为两大类

* 基于用户的协同过滤算法  
* 基于物品的协同过滤算法  

### 基于用户的协同过滤算法

基于用户的协同过滤算法主要包括两个步骤:  
（1）找到和目标用户兴趣相似的用户集合  
（2）找到这个集合中的用户喜欢的，且目标用户没有听说过的物品推荐给目标用户。  

> 步骤（1）的关键就是计算两个用户的兴趣相似度。这里，协同过滤算法主要利用行为的相似度计算兴趣的相似度。给定用户u和用户v，令N(u)表示用户u曾经有过正反馈的物品集合，令N(v)为用户v曾经有过正反馈的物品集合。那么，我们可以通过如下的jacceard公式简单地计算u和v的兴趣相似度  
N(u)&&N(v)/N(u)||N(v)  
或者通过余弦相似度计算  
N(u)&&N(v)/sqrt(|N(u)N(v)|)

```
import math
# 余弦相似度
def UserSimilarity(train):
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            W[u][v] = len( train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W
```
#### 实际应用中缺点
该代码对两两用户都用余弦相似度计算相似度。这种方法的时间复杂度是O(|U| * |U|)，这在用户数很大时非常耗时。
>事实上，很多用户相互之间并没有对同样的物品产生过行为，即很多时候N(u)&&N(v)=0.**上面的算法将很多时间浪费在了计算这种用户之间的相似度上**。如果换一个思路，我们可以首先计算出N(u)&&N(v)！=0的用户对，然后去计算。

为此，可以首先**建立物品到用户的倒排表**，对于每个物品都保存着对该物品产生过行为的用户列表。令稀疏矩阵C[u][v]=|N(u)&&N(v)|.  
那么假设用户u和用户v同时属于倒排表中K个物品对应的用户列表，就有C[u][v]=K. 从而，可以扫描到倒排表中每个物品对应的用户列表，将用户列表中的两两用户对应的C[u][v]加1，最终就可以得到所有用户之间不为0的C[u][v]。

```
import math
# 余弦相似度
def UserSimilarity(train):
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            W[u][v] = len( train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

# 建立物品-用户倒排索引列表
def UserSimilarity(train):
    # 物品-用户倒排
    item_users = dict()
    for u, items in train.keys():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    
    # 计算物品-用户矩阵
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1
    
    # 计算
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    
    return W
```

得到用户之间的兴趣相似度后，UserCF算法会给用户推荐和他兴趣最相似的K个用户喜欢的物品。  
如下的公式度量了UserCF算法中用户u对物品i的感兴趣程度：
>公式不显示 

其中，S(u,K)包含和用户u兴趣最接近的K个用户，N(i)是对物品i有过行为的用户集合，Wuv是用户u和用户v的兴趣相似度，Rvi代表用户v对物品i的兴趣，因为使用的是单一行为的隐反馈数据，所以所有的Rvi=1

UserCf 的推荐算法实现伪代码
```
# UserCF推荐算法
def Recommend(user, train, W):
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[u].items(), key=itemgetter(1), reverse=True)[0:K]:
        for i, rvi in train[v].items():
            if i in interacted_items:
                continue
            rank[i] += wuv * rvi 
    return rank
```

**用户相似度计算的改进**
前面计算用户兴趣相似度的最简单的公式（余弦相似度公式），但这个公式过于粗糙，本节将讨论如何改进该公式来提高UserCF的推荐性能。

举个例子，以图书为例，如果两个用户都购买过《新华词典》，这丝毫不能说明他们兴趣相似，因为绝大多数中国人小时候都买过《新华词典》。但如果两个用户都买过《数据挖掘导论》，那可以认为他们的兴趣比较相似，因为只有研究数据挖掘的人才会买这本书。**换句话说，两个用户对冷门物品采取同样的行为更能说明他们兴趣的相似度**。

**因此通过1/(log1 + |N(i)|) 来惩罚用户u和用户v共同兴趣列表中热门物品对他们相似度的影响。**

其实就是这本书要是被越多的用户拥有，把这个书的权重就略微低。

将上面的改进的策略，UserCF算法记为User-IIF算法。
```
# 建立物品-用户倒排索引列表, 新增惩罚阈值，User-IIF
def UserSimilarity(train):
    # 物品-用户倒排
    item_users = dict()
    for u, items in train.keys():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    
    # 计算物品-用户矩阵
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                # 主要这改变权重
                C[u][v] += 1/math.log(1+len(users))
    
    # 计算
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    
    return W
```

### 实际在线系统使用的UserCF的例子  
相比基于物品的协同过滤算法(itemCF)， UserCF在目前的实际应用中使用并不多。  
>最著名的使用者是Digg, 在2008年对推荐系统进行了新的尝试。

### 基于物品的协同过滤算法

基于物品的协同过滤算法是目前业界应用最多的算法。  
基于用户的协同过滤算法在一些网站中(如Digg)中得到了应用，但该算法有一些缺点。
* 随着用户数目越来越大，计算**用户兴趣相似度矩阵**将越来越困难，其运算**时间复杂度和空间复杂度**的增长和用户数的增长近似于平方关系。  
* 基于用户的协同过滤很难对推荐结果作出解释  

基于物品的协同过滤算法（简称ItemCF）给用户推荐那些他们之前喜欢的物品相似的物品。比如，该算法会因为你购买《数据挖掘》而给你推荐《机器学习》。不过,ItemCF算法并**不利用物品的内容属性**计算物品之间的相似度，它主要通过**分析用户的行为记录**计算物品之间的相似度。

该算法认为，物品A和物品B具有很大的相似度是因为喜欢物品A的用户大都喜欢物品B。

基于物品的协同过滤算法主要分为两步。  
（1）计算**物品之间的相似度**  
（2）根据**物品的相似度和用户的历史行为**给用户生成推荐列表。

Wij = (|N(i)&&N(j)|)/|N(i)|
分母|N(i)|是喜欢物品i的用户数，而分子是同时喜欢物品i和物品j的用户数。因此，上述公式可以理解为喜欢物品i的用户中有多少比例的用户也喜欢物品j

上述公式虽然看起来很有道理，但是却存在一个问题。如果物品j很热门，很多人喜欢，那么Wij就会很大，接近1。因此，会造成任何物品都会和热门的物品有很大的相似度，这对于致力于挖掘长尾信息的推荐系统来说显然不是一个好的特征。

为了避免推荐出热门的物品，可以用下面的公式。
Wij = (|N(i)&&N(j)|)/sqrt(|N(i)|* |N(j)|)
这个公式**惩罚了物品j的权重**，因此减轻了热门物品会和很多物品相似的可能性。

从上面的定义可以看到，在协同过滤中**两个物品产生相似度是因为它们共同被很多用户喜欢**，也就是说**每个用户都可以通过它们的历史兴趣列表给物品“贡献”相似度**。

这里面蕴含着一个假设，就是每个用户的兴趣都局限在某几个方面，因此如果两个物品属于一个用户的兴趣列表，那么这两个物品可能就属于有限的几个领域，而如果两个物品属于很多用户的兴趣列表，那么它们就可能属于同一个领域，因而有很大的相似度。

ItemCF算法计算物品相似度时可以首先建立用户-物品倒排表
```
import math
# 建立用户-物品倒排索引
def ItemSimilarity(train):
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
    
    # calculate 
    W = dict()
    for i, related_items in C.items():
        for j , cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i]*N[j])

    return W
```

上面程序计算物品相似性的中，对于每个物品集合，我们将里面的物品两两加一，得到一个矩阵。最终将这些矩阵相加得到上面的C矩阵。其中C[i][j]记录了同时喜欢物品i和物品j的用户数。最后，将C矩阵**归一化**可以得到**物品之间的余弦相似度矩阵W**

用户历史上感兴趣的物品越相似的物品，越有可能在用户的推荐列表中获得比较高的排名。  
伪代码如下
```
# 和用户历史上感兴趣的物品越相似的物品，越有可能在用户的推荐列表中获得比较高的排名  
def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j] += pi * wj
    return rank
```

ItemCF的一个优势就是可以提供推荐解释，即利用用户历史上喜欢的物品为现在的推荐结果进行解释。

```
# 带解释的ItemCF算法
def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    # 意思是用户对每一本书有一个感兴趣的概率
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j].weight += pi*wj
            rank[j].reason[i] = pi* wj
    return rank
```

**用户活跃度对物品相似度的影响**  
在协同过滤中两个物品产生相似度是因为它们共同出现在很多用户的兴趣列表中。换句话说，**每个用户的兴趣列表都对物品的相似度产生贡献**。
但是，并不是每个用户的贡献都是相同的。  
举个例子，有个用户开书店的，买了当当网80%的书准备自己卖，那么，他的购物车里包含当当网80%的书。假设当当网100万书，也就是说他买了80万本。从前面对ItemCF的讨论可以看到，这意味着因为存在这么一个用户，有80万本书两两之间就产生了相似度，也就是说，内存里即将诞生一个80万乘80万的稠密矩阵。

另外可以看到，这个用户虽然活跃，但是买这些书并非都出于自身的兴趣，而且这些书覆盖了当当网图书的很多领域，所以这个用户对于他所购买书的两两相似度的贡献应该远远小于一个只买了十几本自己喜欢的书的文学青年。

John S.Breese在论文中提出了一个称为**IUF（Inverse User Frequence），即用户活跃度对数的倒数的参数**，他也认为活跃用户对物品相似度的贡献应该小于不活跃的用户，他提出应该增加IUF参数来修正物品相似度的计算公式

```
import math
def ItemSimilarity(train):
    #calculate co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1/math.log(1+len(items) * 1.0)
    
    # calculate finial similarity matrix W
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W[i][j] = cij /math.sqrt(N(i)*N(j))
    return W
```

**上面的算法记为Item-IUF**

物品相似度的归一化，对ItemCF的相似度矩阵按最大值归一化，可以提高推荐的准确率。  
归一化的好处不仅仅在于**增加推荐的准确度**，还可以**提高推荐的覆盖率和多样性**

#### UserCF算法和ItemCF算法的推荐原理

UserCF给用户推荐那些和他有**共同兴趣爱好的用户喜欢的物品**，而ItemCF给用户推荐那些**和他之前喜欢的物品类似的物品**。从这个算法的原理可以看到，UserCF的推荐结果着重于反映和用户兴趣相似的**小群体的热点**，而ItemCF的推荐结果着重于**维系用户的历史兴趣。**  
换句话说，UserCF的推荐更社会化，反映了用户所在的小型兴趣群体中物品的热门程度，而ItemCF的推荐更加个性化，反映了用户自己的兴趣传承。

### 参考文献
1. 项亮. 推荐系统实践[M]. 北京: 人民邮电出版社, 2012.