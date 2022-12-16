import tensorflow as tf

tensor = tf.constant([3, 4, 5])
tensor2 = tf.constant([6, 7, 8])

tensor3 = tf.constant([[1, 2], [3, 4]])
tensor4 = tf.constant([[5, 6], [7, 8]])

tensor5 = tf.zeros([2, 2, 3])

print(tf.add(tensor, tensor2))
print(tf.subtract(tensor, tensor2))
print(tf.divide(tensor, tensor2))
print(tf.multiply(tensor, tensor2))
print(tf.matmul(tensor3, tensor4))
print(tensor5)

w = tf.Variable(1.0)
print(w)
