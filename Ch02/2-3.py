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

# 带解释的ItemCF算法
def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            #rank[j] += pi * wj
            rank[j].weight += pi*wj
            rank[j].reason[i] = pi* wj
    return rank