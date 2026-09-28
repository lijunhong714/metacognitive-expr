import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "akt"
        self.n_question = n_question # 这里的n_question其实是 concept数量，所以应该是n_concept才对，通常取值远小于n_pid
        self.dropout = dropout
        self.kq_same = kq_same # 1 => query 与 key 共用同一个线性投影 k_linear（省掉一半投影参数）；0 => query 另用 q_linear。它与“自注意力/交叉注意力”无关：同一个值会传给编码层和检索层的所有 block
        self.n_pid = n_pid # 这里的n_pid是题目的数量,对于不含有question_id 的数据集，例如assist2015,这个取值为0
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        '''
            论文中的
            q_data: (batch_size,seq_len), 取值[0,C-1]
            target: (batch_size,seq_len),取值{0,1}
        '''
        q_embed_data = self.q_embed(q_data)  # (batch_size,seq_len,d)，对应论文xt计算过程中的cct部分
        # 嵌入过程参数量C*D
        if self.separate_qa:
            qa_data = q_data + self.n_question * target # (batch_size,seq_len)，取值[0,2C-1],其中答错的部分取值[0,C-1]，答对的部分取值[C,2C-1]；
            qa_embed_data = self.qa_embed(qa_data) # (batch_size,seq_len,d)
            # 嵌入过程的参数量2*C*D
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            # target从(batch_size,seq_len)变为(batch_size,seq_len,d)
            qa_embed_data = self.qa_embed(target)+q_embed_data # 这是论文中提到的分支，事实上一直以来默认都走的是seperate_qa=0这条通路。
            # 嵌入过程计算量 target嵌入的：2*D,q嵌入的C*D,共（C+2)*D；
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False):
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0: # have problem id
            # (batch_size,seq_len,D) 参数量为C*D
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化,对应公式xt计算公式中的d_ct部分
            # (batch_size,seq_len,1) 参数量为（Q+1)*1近似于Q*1
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # 对应公式中的xt
            # 总参数量=cct的参数量+d_ct的参数量+u的参数量 = C*D+C*D+Q*1=2CD+Q,与论文一致。(batch_size,seq_len,D)
            
            qa_embed_diff_data = self.qa_embed_diff(
                target)  # 对应公式中的f(ct,rt)，concept-response变量，输出(batch_size,seq_len,D)。表声明为(2C+1)*D，但 target 只取{0,1}，实际只用到前 2 行 => 生效参数量 2*D
            if self.separate_qa:
                # (batch_size,seq_len,D)
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  
                # 参数量：2*C*D(qa_embed，(2C+1) 行基本全用) + Q*1(difficult_param) + 2*D(qa_embed_diff 实际生效行)
            else:
                # 走这条通路,求出的值对应公式中的yt
                # (batch_size,seq_len,D)
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  
                # 参数量 (C+2)*D(q_embed 的 C*D + qa_embed 的 2*D) + Q*1(difficult_param) + 2*D(qa_embed_diff 生效行) + C*D(q_embed_diff)
                # 这里 f(ct,rt) 用的是 qa_embed_diff 与 q_embed_diff 之和
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2 # rasch部分loss
        else:
            c_reg_loss = 0. # 对于不含有problem_id的数据集，直接将损失值置0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        # 输入的分别为最终问题最终嵌入xt、交互最终嵌入yt、问题复杂度表征u。
        # shape分别为（B,L,D),(B,L,D),(B,L,1)
        # 输出的部分对应知识检索模块的输出ht
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data) # (batch_size,seq_len,d_model)
        # ht和问题嵌入共同构成预测层的输入
        concat_q = torch.cat([d_output, q_embed_data], dim=-1) # (batch_size,seq_len,d_model+embed_l)
        output = self.out(concat_q).squeeze(-1) # self.out 最后一层是 Linear(256,1)，故 concat_q(B,L,2D) -> (B,L,1) -> squeeze(-1) -> (batch_size,seq_len)
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q

# Encoder和知识检索（knowledge retriever)模块代码
# 其中Encoder层主要是自注意力，知识检索为交叉注意力机制
class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data) # yt^ 这里是自注意力机制
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        # 创建一个上三角矩阵 当k=0时，保留主对角线及其上方，k=1时保留主对角线上方（不包含主对角线）
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        # 反转上面的nopeek_mask,得到一个下三角矩阵，当k=0时，不包括主对角线，只能查看0-t-1时刻的记录，k=1时，包括主对角线，能够查看0-t时刻的记录
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.emb_type = emb_type
        if emb_type.endswith("avgpool"):
            # pooling
            #self.pool =  nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            pool_size = 3
            self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            # linear
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas) # 距离衰减因子，对应论文公式1中的部分，初始为随机值，后续可修改
            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):
        '''
        q,k,v (batch_size,seq_len,d_model)
        '''

        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            # v = v.transpose(1,2)
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)#.contiguous().view(bs, -1, self.d_model)
        elif self.emb_type.endswith("linear"):
            # v = v.transpose(1,2)
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)
        elif self.emb_type.startswith("qid"):
            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # (batch_size,seq_len,d_model)->(batch_size,seq_len,n_heads,d_feature)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_model (batch_size,n_heads,seq_len,d_feature)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next 也就是论文中的单调注意力机制
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff) # (batch_size,nheads,seq_len,d_feature)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model) # (batch_size,seq_len,d_model)

        output = self.out_proj(concat) # (batch_size,seq_len,d_model)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen (batch_size,n_heads,seq_len,seq_len) 也就是注意力分数scores
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    
    # 构造位置距离矩阵
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device) # 生成[0,1,...,seq_len-1] (1,seq_len)->（seq_len,seq_len) 得到[[0,1,...seq_len-1],[0,1,...seq_len-1],...]
    x2 = x1.transpose(0, 1).contiguous() # 求转置得到[[0,0,...0],[1,1,....,1],...,[seq_len-1,...seq_len-1]]，最后一行为 seq_len-1

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32) # mask标记为0，赋值为负无穷
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device) # 结果和上一步一样，如果mask标记为0，赋值为0
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl，对于最后一个维度进行前缀和计算
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1  最后一个维度直接求和,数值上=distcum_scores[-1]
        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff == None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    else:
        # pdiff:问题复杂度(batch_size,seq_len,1) -> (batch_size,1,seq_len,1)->(batch_size,n_heads,seq_len,1)
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32) # 只要mask矩阵中标定为0的，全部搞成负无穷
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v) #(batch_size,n_heads,seq_len,seq_len) @ (batch_size,n_heads,seq_len,d_k)，此处 v 的末维是“每头维度” d_k = d_model/n_heads，不是 d_model
    # import sys
    # sys.exit()
    return output # (batch_size,n_heads,seq_len,d_k)，每头维度；回到 MultiHeadAttention 后 view 拼头才还原成 d_model

# 消融实验,可学习位置编码
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model) #生成符合正态分布的张量(max_len,d_model)
        pe = pe.unsqueeze(0) # (1,max_len,d_model)
        self.weight = nn.Parameter(pe, requires_grad=True) # pe这个张量的权重需要随训练不断更新

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature) ，也就是第1维度的max_len 中取前seq_len 项返回。

# 消融实验，余弦位置编码,是Attention is All You Need中采用的方法
class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() #（maxlen,1) [[0],[1],....[max_len-1]]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
