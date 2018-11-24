import os
from common.config import GPU

import numpy

parent_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(parent_dir, '../../1_data')

test_data_file = os.path.join(test_path, 'train_data.npy')
test_label_file = os.path.join(test_path, 'train_label.npy')

train_path = os.path.join(parent_dir, '../../gen_numpy')
train_num = 6000
img_dim = (1, 28, 28)
img_size = 784

def load_katakana_arg(normalize=True, flatten=False, shuffle=True):

    """カタカナデータセットの読み込み(アノテーション付き)

    Parameters
    ----------
    normalize: 画像のピクセル値を0
    .0
    ~1.0
    に正規化する

    flatten: 画像を一次元配列に平にするかどうか
    shuffle: データを社フルするかどうか
    devide: データをどの程度訓練データに割るか、5の場合、4/5が訓練データで、1/5がテストデータ

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    # テストデータの読み込み
    test_data = numpy.load(str(test_data_file))
    test_label = numpy.load(str(test_label_file))

    # 訓練データの読み込み

    list_data  = []
    list_label = []

    def load_train_data(num):
        data_path = os.path.join (train_path, '{:02d}_data.npy'.format(num) )
        label_path = os.path.join (train_path, '{:02d}_label.npy'.format(num) )
        in_data = numpy.load(str(data_path))
        in_label = numpy.load(str(label_path))
        list_data.extend(in_data)
        list_label.extend(in_label)

    for i in range(15):
        load_train_data(i)

    # Numpyへ変換
    list_data = numpy.array(list_data)
    list_label = numpy.array(list_label)

    # シャルフルするかどうか
    if shuffle:
            indexes = numpy.arange(len(list_data))
            numpy.random.shuffle(indexes)

            data = list_data[indexes]
            label = list_label[indexes]

    if GPU:
        # GPUに転送
        import cupy
        data = cupy.array(data)
        label = cupy.array(label)
        test_data = cupy.array(test_data)
        test_label = cupy.array(test_label)

    if normalize:
        data /= 255.0
        test_data /= 255.0

    if flatten:
        data.reshape(-1, img_size)
        test_data.reshape(-1, img_size)

    x_train = data
    t_train = label
    x_test = test_data
    t_test = test_label

    return (x_train, t_train), (x_test, t_test)


def t_main():
    (x_train, t_train), (x_test, t_test) = load_katakana_arg()
    print('x_train', x_train.shape, 't_test', t_test.shape)
    print('x_test', x_train.shape, 't_test', t_test.shape)

    print('-------------- x_test-------------- ')
    print(t_train[:5])

    print('-------------- t_test-------------- ')
    print(t_test[:5])


if __name__ == '__main__':
    t_main()

