from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# load the data
# image size 28x28 px, output 1 int
# 60,000 training data, 10,000 test data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# let there be model
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_pred = tf.nn.softmax(tf.matmul(x, w) + b)


# define criteria
cross_entropy = -tf.reduce_mean(y_true * tf.log(y_pred)) * 1000.0
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2000):
	batch_xs, batch_ys = mnist.train.next_batch(500)
	if i%100 == 0:
		train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch_xs, y_true: batch_ys})
		print("step %d, training accuracy %.3f"%(i, train_accuracy))
	sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

# evaluate
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))