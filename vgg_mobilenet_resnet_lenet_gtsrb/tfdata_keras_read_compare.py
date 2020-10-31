import tensorflow as tf
import numpy as np
import time

print(tf.__version__)

# 本文将对比tf.data与tf.keras中keras读数据方式下那种速度快，具体有三点：
# tf.data与keras生成器读数据速度对比
# tf.data包装后的keras生成器与原始生成器速度对比
# model.fit 与 model.fit_generator分别使用以上数据的实验

# 1、三种数据读取方式对比

# 1.1、准备本文所用数据
(train_x,train_y),(test_x,test_y) = tf.keras.datasets.fashion_mnist.load_data()
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
train_x=np.expand_dims(train_x,-1)      # keras生成器读数据要求输入形状是rank=4
test_x=np.expand_dims(test_x,-1)

# 1.2、准备tf.data数据
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))

train_ds = train_ds.shuffle(buffer_size=1000).batch(256).prefetch(buffer_size=1000).repeat()    # 训练数据会一直重复读
test_ds = test_ds.batch(256).prefetch(buffer_size=1000)     # 测试数据只读一遍，所以没有加repeat，也可以加repeat(1)

# 检查数据
for data,label in test_ds.take(1):
    pass
print(data.shape,label.shape)
np.testing.assert_array_almost_equal(data,test_x[:256,...])     # 不返回报错信息表示数据相等
np.testing.assert_array_almost_equal(label,test_y[:256])
print(train_ds)

# 1.3、keras生成器读数据方式
gen = tf.keras.preprocessing.image.ImageDataGenerator()     # 不做任何数据预处理

train_flow=gen.flow(train_x,train_y,batch_size=256,shuffle=True)        # 与tf.data中batch相同大小，并且shuffle
test_flow=gen.flow(test_x,test_y,batch_size=256,shuffle=False)

# 检查数据
data,label= next(test_flow)
np.testing.assert_array_almost_equal(data,test_x[:256,...])     # 不返回报错信息表示数据相等
np.testing.assert_array_almost_equal(label,test_y[:256])
print(train_flow)

# 1.4、tf.data包装keras生成器
gen = tf.keras.preprocessing.image.ImageDataGenerator()
wrap_train_ds = tf.data.Dataset.from_generator(lambda:gen.flow(train_x,train_y,batch_size=256,shuffle=True),
    output_types=(tf.uint8, tf.uint8),
    output_shapes = ([None,28,28,1],[None])
)
wrap_test_ds = tf.data.Dataset.from_generator(lambda:gen.flow(test_x,test_y,batch_size=256,shuffle=False),
    output_types=(tf.uint8, tf.uint8),
    output_shapes = (tf.TensorShape([None,28,28,1]),tf.TensorShape([None]))#tf.TensorShape可以不用
)

# 检查数据
for data,label in wrap_test_ds.take(1):
    pass
print(data.shape,label.shape)
np.testing.assert_array_almost_equal(data,test_x[:256,...])     # 不返回报错信息表示数据相等
np.testing.assert_array_almost_equal(label,test_y[:256])
print(wrap_train_ds)

# 1.5、有了三种数据类型开始比较速度
default_timeit_steps = 5000

def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i%50 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} samples/s".format(256*steps/duration))

timeit(train_ds)

timeit(train_flow)

timeit(wrap_train_ds)
# 对比结论
# 显然tf.data是最快的，wrap后的生成器最慢，我们肯定是要用tf.data的。
# 关于wrap后比原始keras读数据的方式慢的原因，可能是因为这个生成器有问题，具体不再深究，所以我们就直接用tf.data了。
# keras的generator读取方法速度慢可能也与tensorflow本身有关，
# 现在tensorflow2.1rc对fit_generator,predict_generator都有做修改，并且有关generator的api都会弃用，fit便直接支持生成器。


# 2.对tf.data进行改进
# 2.1 使用AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
train_ds = train_ds.shuffle(buffer_size=1000).batch(256).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
test_ds = test_ds.batch(256).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

timeit(train_ds)
# 提升看不明显，在其它数据上有试，效果会比自己随意设定的快


# 2.2 使用cache
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))

train_ds=train_ds.cache().shuffle(buffer_size=1000).batch(256).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
test_ds = test_ds.batch(256).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

timeit(train_ds)
# 这速度的提升是惊人的，这些操作能使读取速度得到提升，但是提升多少依数据类型和其它参数的改变而改变


# 2.3 map的使用
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
def transfer(value1,value2):
    return value1,value2    # 什么操作都不加，只是为了配合map来使用

train_ds = train_ds.cache().shuffle(buffer_size=1000).map(transfer, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                           .batch(256).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()

timeit(train_ds)

# 需要对数据做处理时要要用到map.
#
# 说明： 关于对shuffle,cache,batch,map,prefeach,repeat的顺序，排列组合情部很多，产生数据是相同的，
# 但在数据最后一部分不够一个batch size的情况下有些许不同，但对训练没太大影响，
# 测试数据只要全部读取就好。关于速度的影响，推荐使用上边代码的顺序。

# 3、对第一节中三种数据分别训练模型（fit,fig_generator的使用）
def get_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28,1)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model= get_model()
start = time.time()
model.fit(train_ds,
         steps_per_epoch=train_x.shape[0]//32,
         epochs=5)
print("It took {} seconds".format(time.time() - start))

model= get_model()
start = time.time()
model.fit_generator(train_flow,
                   steps_per_epoch=train_x.shape[0]//32,
                   epochs=5)
print("It took {} seconds".format(time.time() - start))

model= get_model()
start = time.time()
model.fit(wrap_train_ds,
         steps_per_epoch=train_x.shape[0]//32,
         epochs=5)
print("It took {} seconds".format(time.time() - start))

#可以看出，tf.data训练要更快，并且精度高一些（这个有点不太明白，
# 后其文章会对fit,fit_generator做更多说明）