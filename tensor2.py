import tensorflow as tf

height = 170
foot = 260

a = tf.Variable(0.1)
b = tf.Variable(0.2)


def loss():
    return tf.square(foot-(height*a+b))


opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(loss, var_list=[a, b])
    print(a.numpy(), b.numpy())
