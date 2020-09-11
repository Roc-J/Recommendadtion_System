import random

# 隐语义模型负样本采用过程
def RandomSelectNegativeSample(self, items, items_pool):
    ret = dict()
    # items是用户已经有过行为的物品的集合
    for i in items.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(items)*3):
        item  = items_pool[random.randint(0, len(items_pool)-1)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret


def LatentFactorModel(user_items, F, N, alpha, lambda):
    [P, Q] = initModel(user_items, F)
    for step in range(0, N):
        for user, items in user_items.items():
            samples = RandomSelectNegativeSample(items)
            for item, rui in samples.items():
                eui = rui - Predict(user, item)
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - lambda * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - lambda * Q[item][f])
        alpha *= 0.9

def Recommend(user, P, Q):
    rank = dict()
    for f, puf in P[user].items():
        for i, qfi in Q[f].items():
            if i not in rank:
                rank[i] += puf * qfi
    return rank