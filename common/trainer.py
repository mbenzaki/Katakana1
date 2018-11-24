# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.optimizer import *
import math

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True,
                 early_stop_count=2):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.early_stop_count = early_stop_count

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(math.ceil(self.train_size / mini_batch_size), 1)
        #self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)

        self.best_test_loss = np.finfo(np.float64).max
        self.worse_count = 0

        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list =[]
        self.test_loss_list = []


    def train_epoch(self):
        indexes = np.arange(self.train_size)
        np.random.shuffle(indexes)

        if self.verbose:
            print('Epoch:', self.current_epoch+1, end=" ", flush=True)

        # Iterationのループ
        for self.current_iter in range(self.iter_per_epoch):
            worse_count = 0

            start = self.batch_size * self.current_iter
            end = self.batch_size * (self.current_iter+1)

            batch_mask = indexes[start:end]

            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            loss = self.network.loss(x_batch, t_batch)

            if self.verbose:
                if (self.current_iter % 2) == 0:
                    print('.', end="", flush=True)

        # end of Iterationのループ

        train_acc, train_loss = self.network.accuracy_and_loss(self.x_train, self.t_train)
        test_acc, test_loss = self.network.accuracy_and_loss(self.x_test, self.t_test)

        if self.verbose:
            print() #　改行
            print('Epoch={}, Train Acc={}, Train Loss={}, Test Acc={}, Test Loss={}'.
                  format(self.current_epoch + 1, train_acc, train_loss, test_acc, test_loss) )

        # Early Stopping 用の計算
        if len(self.test_loss_list) > 0 and (test_loss > self.test_loss_list[-1]):
            self.worse_count += 1
            print('Test Loss did not improved best= {} current {}'.
                  format(self.best_test_loss, test_loss))
        else:
            #今までよりもval_losが良ければ、保存する。
            if test_loss < self.best_test_loss:
                if GPU: #このままではこれらの値がstrに変更できない
                    train_acc = np.float(train_acc)
                    train_loss = np.float(train_loss)
                    test_acc = np.float(test_acc)
                    test_loss = np.float(test_loss)
                file_name = 'epoch_{:03d}_train_acc_{:.2f}_train_loss_{:.2f}_test_acc_{:.2f}_test_loss_{:.2f}.pkl'.\
                    format(self.current_epoch, train_acc, train_loss, test_acc, test_loss)
                if self.verbose:
                    print('Test Loss improved from {} to {}\nSave weight to {}'.
                          format(self.best_test_loss, test_loss, file_name))
                self.network.save_params(file_name)
                self.best_test_loss = test_loss
            else:
                if self.verbose:
                    print('Test Loss did not improved best={} current={}'.
                          format(self.best_test_loss, test_loss))

            self.worse_count = 0

        if self.worse_count > self.early_stop_count:
            return False

        self.train_acc_list.append(train_acc)
        self.train_loss_list.append(train_loss)
        self.test_acc_list.append(test_acc)
        self.test_loss_list.append(test_loss)

        return True

    def train(self):
        for self.current_epoch in range(self.epochs):

            if not self.train_epoch():
                print('Early Stopping at epoch = ', self.current_epoch+1)
                break

        if self.verbose:
            print('Epoch end ', self.current_epoch + 1)

        test_acc, test_loss  = self.network.accuracy_and_loss(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
            print("test loss:" + str(test_loss))

        return self.train_acc_list, self.train_loss_list, self.test_acc_list, self.test_loss_list
