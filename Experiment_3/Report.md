# 信息检索第三次实验报告
## 201800130083 数据18 刁龙飞
--- 
### 实验目的：
- 理解老师给出的代码
- 补全代码，完成MAP、MRR和NDCG三种评价指标
- 在提供的仿真数据集上完成三种指标的评价
### 实验环境：
- 硬件环境：个人PC机
- 软件环境：Jupyter notebook, Spyder Python3.6   
---
### 实验流程：
#### 对于已给出的代码的理解：
最原始的数据集包括tweets.txt和qrels2014.txt，表示所有的tweets和基于tweets的查询结果。在qrels2014.txt文件中每一行按照空格分割有四列，分别表示查询编号、类型、相关tweet编号、关联值。关联值有0，1，2三种，会在计算NDCG评价指标时用到。   
>171 Q0 307390255743528960 2   
171 Q0 307431003394281472 2   
171 Q0 297987456635117569 0   
171 Q0 307496547812270080 2   
171 Q0 307364469183483906 2   
171 Q0 307381804225163264 1   
171 Q0 298537912893595649 0   
171 Q0 307600151294382080 2   

qrels.txt是对quels2014.txt处理而来，剔除了quels2014.txt中相关tweet编号不在tweets.txt中的行。我们将该数据集用作事实上的相关性标准，对结果进行评价。

result.txt为一个仿真的结果，意在模拟某个信息检索系统对于相关quel得到的相关性文档的结果。我们的目的是采用MAP、MRR和NDCG三种评价指标对该结果进行评价。

已给出的代码包括两个文件，第一个文件process.py用于对原始数据集tweets.txt和qrels2014.txt进行预处理及模拟仿真，并得到qrels.txt和result.txt两个结果。evalution.py用于补全代码，对result.txt的结果加以评价，输出评价值。

下面补充的函数参数说明：
>qrels_dict = {query_id:{doc_id:gain, doc_id:gain, ...}, ...}   
>test_dict = {query_id:[doc_id, doc_id, ...], ...}   
>k为阈值，即取返回到的结果的前k个进行评价。用户往往不关注靠后的检索结果。默认k=100

#### 使用MAP指标：
对于每个查询，遍历r从1到result对应的文档长度。将前r个结果中的正确率求平均，得到Average Precision。最后将得到的所有的Average Precision再求平均，得到MAP。
```python
def MAP_eval(qrels_dict, test_dict, k = 100):
    #每个quel的precision的平均
    map_list = []
    for quel in qrels_dict:
        k_list = []
        for r in range(len(test_dict[quel][:])):
            related_num = 0
            for i in test_dict[quel][:r]:
                if i in qrels_dict[quel]:
                    related_num += 1
            k_list.append(related_num/len(test_dict[quel][:]))
        map_list.append(np.mean(np.array(k_list)))
    return np.mean(np.array(map_list))
```

#### 使用MRR指标：
对于每个查询，只关注与它相关的第一个文档，找到这个文档在查询结果中的位置，对该位置取倒数。需要注意的是，由于python的位置编号从0开始，这里需要+1。
```python
def MRR_eval(qrels_dict, test_dict, k = 100):
    K_list = []
    for quel in qrels_dict:
        First_doc = list(qrels_dict[quel].keys())[0]
        K = test_dict[quel].index(First_doc)
        K_list.append(K)
    return np.mean(1/(np.array(K_list)+1))
```

#### 使用NDCG指标：
对于每个查询，首先对于返回的结果文档集前k个计算DCG。我采用的数据结构是numpy提供的array类型。由于返回的结果文档集可能是不相关的，此时的相关性就是0。每次查询得到一个DCG，为了将其归一化，我们需要求出理想状况下的DCG即IDCG。   

由于gain的取值只有0,1,2，而相关的只有1,2，我们将与该查询相关的文档按照gain进行排序。得到，在给定相关的文档集上计算IDCG，使用IDCG对DCG进行归一化得到NDCG。

最后所有查询的NDCG求平均。


参考资料：https://www.cnblogs.com/by-dream/p/9403984.html
```python
def NDCG_eval(qrels_dict, test_dict, k = 100):
    NDCG = []
    for quel in qrels_dict:
        rel = []
        i = np.arange(1,k+1)
        for j in test_dict[quel][:k]:
            try:
                rel.append(qrels_dict[quel][j])
            except KeyError:
                rel.append(0)        
        i = np.arange(1,min(len(test_dict[quel][:]),k)+1)
        DCG_list = np.array(rel)/np.log2(i+1)#不能是0
        DCG = np.sum(DCG_list)
        sorted_list = [i for i in list(qrels_dict[quel].keys()) if qrels_dict[quel][i] == 2] +\
        [i for i in list(qrels_dict[quel].keys()) if qrels_dict[quel][i] == 1]
        
        IDCG_rel = []
        for j in sorted_list:
            IDCG_rel.append(qrels_dict[quel][j])
        I = np.arange(1,len(sorted_list)+1)
        IDCG_list = np.array(IDCG_rel)/np.log2(I+1)#不能是0
        IDCG = np.sum(IDCG_list)
        NDCG.append(DCG/IDCG)
    return np.mean(np.array(NDCG))
```

结果如下：
>MAP = 0.4264352262227257   
MRR = 0.79737012987013   
NDCG = 0.6875441733398155
---
### 实验心得：
通过学习老师的代码有助于改善自己的编程风格。通过这次实验复习了信息检索的相关评价指标，也练习了相关数据结构的使用。