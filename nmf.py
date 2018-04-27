""" Non-negative matrix factorization (tensorflow)"""
# Author: Eesung Kim <eesungk@gmail.com>


import tensorflow as tf
import numpy as np

class NMF:
    """Compute Non-negative Matrix Factorization (NMF)"""
    def __init__(self, max_iter=200, learning_rate=0.01,display_step=10, optimizer='mu', initW=False):

        self.max_iter = max_iter
        self.learning_rate= learning_rate
        self.display_step = display_step
        self.optimizer = optimizer

    def NMF(self, X, r_components, learning_rate, max_iter, display_step, optimizer, initW, givenW ):
        m,n=np.shape(X)
        tf.reset_default_graph()
        V = tf.placeholder(tf.float32) 

        initializer = tf.random_uniform_initializer(0,1)

        if initW is False:
            W =  tf.get_variable(name="W", shape=[m, r_components], initializer=initializer)
            H =  tf.get_variable("H", [r_components, n], initializer=initializer)
        else:
            W =  tf.constant(givenW, shape=[m, r_components], name="W")
            H =  tf.get_variable("H", [r_components, n], initializer=initializer)

        WH =tf.matmul(W, H)
        cost = tf.reduce_sum(tf.square(V - WH))
        
        if optimizer=='mu':
            """Compute Non-negative Matrix Factorization with Multiplicative Update"""
            Wt = tf.transpose(W)
            H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
            H_update = H.assign(H_new)

            if initW is False:
                Ht = tf.transpose(H)
                W_new = W * tf.matmul(V, Ht)/ tf.matmul(W, tf.matmul(H, Ht))
                W_update = W.assign(W_new)

        elif optimizer=='pg':
            """optimization; Projected Gradient method """
            dW, dH = tf.gradients(xs=[W, H], ys=cost)
            H_update_ = H.assign(H - learning_rate * dH)
            H_update = tf.where(tf.less(H_update_, 0), tf.zeros_like(H_update_), H_update_)

            if initW is False:
                W_update_ = W.assign(W - learning_rate * dW)
                W_update = tf.where(tf.less(W_update_, 0), tf.zeros_like(W_update_), W_update_)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for idx in range(max_iter):
                if initW is False:
                    W=sess.run(W_update, feed_dict={V:X})
                    H=sess.run(H_update, feed_dict={V:X})
                else:
                    H=sess.run(H_update, feed_dict={V:X})

                if (idx % display_step) == 0:
                    costValue = sess.run(cost,feed_dict={V:X})
                    print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue))

        return W, H

    def fit_transform(self, X,r_components, initW, givenW):
        """Transform input data to W, H matrices which are the non-negative matrices."""
        W, H =  self.NMF(X=X, r_components = r_components, learning_rate=self.learning_rate, 
                    max_iter = self.max_iter, display_step = self.display_step, 
                    optimizer=self.optimizer, initW=initW, givenW=givenW  )
        return W, H

    def inverse_transform(self, W, H):
        """Transform data back to its original space."""
        return np.matmul(W,H)

def main():
    V = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    model = NMF(max_iter=200,learning_rate=0.01,display_step=10, optimizer='mu')
    W, H = model.fit_transform(V, r_components=2, initW=False, givenW=0)
    print(W)
    print(H)
    print(V)
    print(model.inverse_transform(W, H))

if __name__ == '__main__':
    main()