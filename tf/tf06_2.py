import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# Try to find values for W and b to compute Y = W * X + b
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis is X * W + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line with new training data
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b],
            feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]},
        )
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    # Learns best fit W:[ 1.],  b:[ 1.1]
    """
    0 1.2035878 [1.0040361] [-0.00917497]
    20 0.16904518 [1.2656431] [0.13599995]
    ...
    1980 2.9042917e-07 [1.00035] [1.0987366]
    2000 2.5372992e-07 [1.0003271] [1.0988194]
    [6.1004534]
    [3.5996385]
    [2.5993123 4.599964 ]
    """