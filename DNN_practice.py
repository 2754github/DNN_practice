## ライブラリのインポート
# 使用するデータセットの指定
from keras.datasets import mnist
# from keras.datasets import fashion_mnist

 # カテゴリカル化に使う
from keras.utils import np_utils

# データの可視化に使う
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# モデルの構築に使う
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


## データの処理
# データセットの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train.shape)
# print(X_train[0])
# %matplotlib inline
# imshow(X_train[0].reshape(28,28))

# 1次元化
x_train = X_train.reshape(60000, 784)
x_test = X_test.reshape(10000, 784)
# print(x_train.shape)

# 正規化
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255
# print(x_train[0])

# カテゴリカル化
# print(y_train)
y_train_label = np_utils.to_categorical(y_train)
y_test_label = np_utils.to_categorical(y_test)
# print(y_train_label)


## MLPモデルの構築
# モデルの定義
def build_multilayer_perceptron():
    model = Sequential()
    
    # 入力層-隠れ層1
    model.add(Dense(512, input_shape=(784,))) # 512 = 2^9 784 = x_train.shapeのサイズ
    model.add(Activation("relu")) # 活性化関数の種類　relu = ランプ関数
    model.add(Dropout(0.2)) # 20%の確率でドロップアウトさせる
    
    # 隠れ層2
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # 出力層
    model.add(Dense(10)) # 10 = labelの数
    model.add(Activation("softmax")) # 活性化関数の種類　softmax = softmax関数
    
    return model

# モデルのビルド
model = build_multilayer_perceptron()
# model.summary()

# モデルのコンパイル
model.compile(loss="categorical_crossentropy", optimizer=SGD(), metrics=["accuracy"])


## いざ，学習
history = model.fit(x_train, y_train_label,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test_label)
                   )


## 結果
# 精度の確認
score = model.evaluate(x_test, y_test_label, verbose=0)
print("精度は", score[1]*100, "%")

# 予測結果の確認
p=12
predicts = model.predict_classes(x_test)
if predicts[p]==y_test[p]:
    print("僕の予測は {0} で，正解も　{1} でした！やったー！".format(predicts[p], y_test[p]))
else:
    print("僕の予測は {0} で，正解は　{1} でした……すいません！".format(predicts[p], y_test[p]))

print("入力した画像↓")
get_ipython().run_line_magic('matplotlib', 'inline')
imshow(X_test[p].reshape(28,28))


## 間違えたものの可視化
wrongs = []
for i, (x,y) in enumerate(zip(y_test,predicts)):
    if x != y:
        wrongs.append((i,(x,y)))
        
print("{0}個中{1}個間違えました！すいません！".format(len(y_test), len(wrongs)))

print("間違えた画像のリスト↓ (最初の20個)")
get_ipython().run_line_magic('matplotlib', 'inline')
f = plt.figure(figsize=(15,15))
for i ,(index, (label, predict)) in enumerate(wrongs[:20]):
    i += 1
    axes = f.add_subplot(4,5,i)
    axes.set_title("index:{0}, label:{1}, predict:{2}".format(index, label, predict))
    axes.imshow(X_test[index])

