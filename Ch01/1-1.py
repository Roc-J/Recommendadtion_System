import math
# records[i] = [u, i, rui, pui]
# rui 是指用户u对物品i的实际评分，pui是算法预测出的用户u对物品i的评分

def RMSE(records):
    return math.sqrt(sum([(rui-pui)*(rui-pui) for u, i, rui, pui in records]) / float(len(records)))

def MAE(records):
    return sum([abs(rui-pui) for u, i, rui, pui in records]) / float(len(records))

# 计算推荐算法准确率和召回率
def Recommend(user, N):
    ret = []
    # 用户推荐的N个物品i
    pass
    return ret

def precisionRecall(test, N):
    hit, n_recall, n_precision = 0,0,0
    for user, items in test.items():
        rank = Recommend(user, N)
        hit  += len(rank & items)
        n_recall += len(items)
        n_precision += N
    return [hit/(1.0*n_recall), hit/(1.0*n_precision)]

# 基尼系数
def GiniIndex(p):
    j = 1
    n = len(p)
    G = 0
    for item, weight in sorted(p.items(), key=itemgetter(1)):
        G += (2*j - n -1) * weight
    return G / float(n-1)