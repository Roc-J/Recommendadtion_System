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