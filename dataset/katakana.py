import os
import common.np as np
from common.config import GPU

if GPU:
    import cupy
import numpy

dataset_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dataset_dir, '../../1_data')

data_file = os.path.join(data_path, 'train_data.npy')
label_file = os.path.join(data_path, 'train_label.npy')

train_num = 6000
img_dim = (1, 28, 28)
img_size = 784

def load_katakana(normalize=True, flatten=False, shuffle=True, devide=5):

    """カタカナデータセットの読み込み

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
    data = numpy.load(data_file)
    label = numpy.load(label_file)

    # シャルフルするかどうか
    if shuffle:
            indexes = numpy.arange(len(data))
            numpy.random.shuffle(indexes)
            data = data[indexes]
            label = label[indexes]

    if GPU:
        # GPUに転送
        import cupy
        data = cupy.array(data)
        label = cupy.array(label)

    if normalize:
        data /= 255.0

    if flatten:
        data.reshape(-1, img_size)

    x_train = data
    t_train = label
    x_test = None
    t_test = None

    if devide != 0:
        size_of_test = len(data) // devide
        size_of_split = size_of_test * (devide-1)
        x_train = data[:size_of_split]
        t_train = label[:size_of_split]
        x_test = data[size_of_split:]
        t_test = label[size_of_split:]

    return (x_train, t_train), (x_test, t_test)


def t_main():
    (x_train, t_train), (x_test, t_test) = load_katakana()
    print('x_train', x_train.shape, 't_test', t_test.shape)
    print('x_test', x_train.shape, 't_test', t_test.shape)

    print('-------------- x_test-------------- ')
    print(t_train[:5])

    print('-------------- t_test-------------- ')
    print(t_test[:5])


if __name__ == '__main__':
    t_main()

