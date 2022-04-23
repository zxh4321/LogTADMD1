import os.path

from . import DomainAdversarial
import torch.nn as nn
import torch.optim as optim
import torch
from utils.utils import get_center, epoch_time, get_dist, dist2label, get_iter
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from gensim.models import Word2Vec
from sklearn import metrics

class LogTAD(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.emb_dim = options["emb_dim"]
        self.hid_dim = options["hid_dim"]
        self.output_dim = options["out_dim"]
        self.n_layers = options["n_layers"]
        self.dropout = options["dropout"]
        self.bias = options["bias"]
        self.device = options["device"]
        self.weight_decay = options["weight_decay"]
        self.window_size = options["window_size"]
        self.step_size = options["step_size"]
        # 编码器
        self.encoder = DomainAdversarial.DA_LSTM(self.emb_dim, self.hid_dim, self.output_dim, self.n_layers, self.dropout, self.bias).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), weight_decay=self.weight_decay)
        self.alpha = options["alpha"]
        self.max_epoch = options["max_epoch"]
        #最小中心值
        self.eps = options["eps"]
        self.source_dataset_name = options["source_dataset_name"]
        self.target_dataset_name = options["target_dataset_name"]
        self.test_ratio = options["test_ratio"]
        self.loss_mse = nn.MSELoss()
        self.loss_cel = nn.CrossEntropyLoss()
        self.w2v = None
        self.center = None
    # 训练函数 参数为可以迭代的对象，和一个中心点
    def _train(self, iterator, center):
        # encoder 为定义的LSTM模型 设置开启训练
        self.encoder.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            #将循环得到的数据转移到 GPU上
            src = batch[0].to(self.device)
            domain_label = batch[1].to(self.device)
            labels = batch[2]
            # 将梯度置零
            self.optimizer.zero_grad()
            # 传入LSTM需要的数据和学习率
            #output是经过LSTM处理后未经过过线性层变换的结果，但在列的维度上进行了取均值，实现loss最小化
            # y_d 是经过梯度反转后得到的结果，目的实现loss最大化
            output, y_d = self.encoder(src, self.alpha)
            # domain_label.view(-1),中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列
            domain_label = domain_label.view(-1)
            center = center.to(self.device)
            # 均方差
            mse = 0
            """
            未搞明白
             if labels[ind] == 1:
                    mse += (10 - self.loss_mse(val, center))
                else:
                    mse += self.loss_mse(val, center)
            cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))
            loss = mse * 10e4 + cel
            """
            for (ind, val) in enumerate(output):
                # 异常
                if labels[ind] == 1:
                    mse += (10 - self.loss_mse(val, center))
                else:
                    mse += self.loss_mse(val, center)
            cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))
            loss = mse * 10e4 + cel
            loss.backward()

            self.optimizer.step()
            #item()用于在只包含一个元素的tensor中提取值，注意是只包含一个元素，否则的话使用.tolist()
            # 在训练时使用item()能够防止tensor无线叠加导致的显存爆炸
            epoch_loss += loss.item()

            center.cpu()
            src.cpu()
            domain_label.cpu()
            output.cpu()
            y_d.cpu()
        # 返回一个epoch的损失
        return epoch_loss / len(iterator)

    def _evaluate(self, iterator, center, epoch):

        self.encoder.eval()

        epoch_loss = 0

        lst_dist = []

        lst_mse = []
        lst_cel = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch[0].to(self.device)
                domain_label = batch[1].to(self.device)
                labels = batch[2]
                output, y_d = self.encoder(src, self.alpha)
                if i == 0:
                    lst_emb = output
                else:
                    lst_emb = torch.cat((lst_emb, output), dim=0)

                domain_label = domain_label.view(-1)

                center = center.to(self.device)

                mse = 0
                for (ind, val) in enumerate(output):
                    if labels[ind] == 1:
                        mse += (10 - self.loss_mse(val, center))
                    else:
                        mse += self.loss_mse(val, center)

                cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))

                lst_mse.append(mse.detach().cpu().numpy())
                lst_cel.append(cel.detach().cpu().numpy())

                loss = mse * 10e4 + cel

                epoch_loss += loss.item()

                lst_dist.extend(get_dist(output, center))

                src.cpu()
                domain_label.cpu()
                lst_emb.cpu()
                output.cpu()
                y_d.cpu()
        # 在训练初期构造中心点
        if epoch < 10:
            center = get_center(lst_emb)
            print('get center:', center)
            # self.eps 最小中心值 默认值是0.1
            #center中的元素满足两个条件会分别赋值-0.1和0.1
            center[(abs(center) < self.eps) & (center < 0)] = -self.eps
            center[(abs(center) < self.eps) & (center > 0)] = self.eps
            print('new center', center)

        print('\nmse:', np.mean(np.array(lst_mse)))
        print('cel:', np.mean(np.array(lst_cel)))
        return epoch_loss / len(iterator), center, lst_dist

    def train_LogTAD(self, train_iter, eval_iter, w2v):
        # 定义最佳的验证损失为正无穷
        best_eval_loss = float('inf')
        # 训练 epoch数
        for epoch in tqdm(range(self.max_epoch)):

            if epoch == 0:
                # 在第一次训练的时候。对于center进行初始化，根据隐藏层的大小进行0初始化
                center = torch.Tensor([0.0 for _ in range(self.hid_dim)])
            # 当训练回合大于9时更新center
            if epoch > 9:
                center = fixed_center
            start_time = time.time()
            # 每一个epoch的训练损失
            train_loss = self._train(train_iter, center)
            # 分别返回的是验证损失，中心点，output到中心点的距离
            eval_loss, center, _ = self._evaluate(eval_iter, center, epoch)

            if epoch == 9:
                fixed_center = center

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)


            # 当验证损失小于最优的验证损失并且当训练的epoch>=9时，对最优验证损失重新赋值
            if eval_loss < best_eval_loss and epoch >= 9:
                best_eval_loss = eval_loss
                torch.save(self.encoder.state_dict(), f'{self.source_dataset_name}-{self.target_dataset_name}.pt')

                self.center = fixed_center.cpu()
                pd.DataFrame(fixed_center.cpu().numpy()).to_csv(f'{self.source_dataset_name}-{self.target_dataset_name}_center.csv')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.10f}')
            print(f'\t Val. Loss: {eval_loss:.10f}')
        self.w2v = w2v
        w2v.save(f'{self.source_dataset_name}-{self.target_dataset_name}_w2v.bin')

    def load_model(self):

        self.w2v = Word2Vec.load(f'{self.source_dataset_name}-{self.target_dataset_name}_w2v.bin')
        self.encoder.load_state_dict(torch.load(f'{self.source_dataset_name}-{self.target_dataset_name}.pt'))
        self.encoder.to(self.device)
        self.center = torch.Tensor(
            pd.read_csv(f'{self.source_dataset_name}-{self.target_dataset_name}_center.csv', index_col=0).iloc[:, 0])

    def _test(self, iterator):
        self.encoder.eval()

        y = []
        lst_dist = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch[0].to(self.device)
                label = batch[2]
                output, _ = self.encoder(src, self.alpha)
                for j in label:
                    y.append(int(j))
                lst_dist.extend(get_dist(output, self.center))
                src.cpu()
        return y, lst_dist
#获得最佳的超球体半径
    def get_best_r(self, iterator, steps=100):
        y, lst_dist = self._test(iterator)
        df = pd.DataFrame()
        df['label'] = y
        df['dist'] = lst_dist
        print(df.groupby(['label']).describe())
        mean_normal = np.mean(df['dist'].loc[df['label'] == 0])
        mean_abnormal = np.mean(df['dist'].loc[df['label'] == 1])
        step_len = (mean_abnormal - mean_normal)/steps
        best_r = 0
        best_auc = -1
        R = mean_normal
        for i in range(steps):
            y_pre = dist2label(lst_dist, R)
            # 通过预测值和实际值计算AUC面积
            auc = metrics.roc_auc_score(y, y_pre)
            #如果AUC面积大于best_auc面积，此时的半径和AUC值都是最佳的
            if auc > best_auc:
                best_r = R
                best_auc = auc
            R += step_len
        return best_r, best_auc

    def get_r_from_val(self, val_df):
        X = list(val_df.Embedding)
        X_new = []
        for i in X:
            temp = []
            for j in i:
                temp.extend(j)
            X_new.append(np.array(temp).reshape(self.window_size, self.emb_dim))
        y_d = list(val_df.target.values)
        y = list(val_df.Label.values)
        X = torch.tensor(X_new, requires_grad=False)
        y_d = torch.tensor(y_d).reshape(-1, 1).long()
        y = torch.tensor(y).reshape(-1, 1).long()
        iterator = get_iter(X, y_d, y)
        R, auc = self.get_best_r(iterator)
        return R, auc

    def testing(self, test_normal_df, test_abnormal_df, r, target=0):
        X = list(test_normal_df.Embedding.values[::int(1 / self.test_ratio)])
        X.extend(list(test_abnormal_df.Embedding.values[::int(1 / self.test_ratio)]))
        X_new = []
        for i in tqdm(X):
            temp = []
            for j in i:
                temp.extend(j)
            X_new.append(np.array(temp).reshape(self.window_size, self.emb_dim))
        y_d = list(test_normal_df.target.values[::int(1 / self.test_ratio)])
        y_d.extend(list(test_abnormal_df.target.values[::int(1 / self.test_ratio)]))
        y = list(test_normal_df.Label.values[::int(1 / self.test_ratio)])
        y.extend(list(test_abnormal_df.Label.values[::int(1 / self.test_ratio)]))
        X_test = torch.tensor(X_new, requires_grad=False)
        y_d_test = torch.tensor(y_d).reshape(-1, 1).long()
        y_test = torch.tensor(y).reshape(-1, 1).long()
        test_iter = get_iter(X_test, y_d_test, y_test)
        y, lst_dist = self._test(test_iter)
        y_pred = dist2label(lst_dist, r)
        if target:
            print(f'Testing result for {self.target_dataset_name}:\n')
        else:
            print(f'Testing result for {self.source_dataset_name}:\n')

        print('Accuracy:', metrics.accuracy_score(y, y_pred))
        # digits 格式化输出浮点值的位数
        #classification_report 构建显示主要分类指标的文本报告
        print(metrics.classification_report(y, y_pred, digits=5))
