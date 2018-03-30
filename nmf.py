""" Non-negative matrix factorization (tensorflow)
"""
# Author: Euisung Kim <eesungk@gmail.com>


import tensorflow as tf
import numpy as np

class NMF:
    """Compute Non-negative Matrix Factorization (NMF)"""
    def __init__(self, r_components=None, max_iter=200, learning_rate=0.01,display_step=10, optimizer='mu'):
        self.r_components = r_components
        self.max_iter = max_iter
        self.learning_rate= learning_rate
        self.display_step = display_step
        self.optimizer = optimizer

    def NMF(self, X, r_components, learning_rate, max_iter, display_step, optimizer ):
        m,n=np.shape(X)
        tf.reset_default_graph()
        V = tf.placeholder(tf.float32) 

        initializer = tf.random_uniform_initializer(0,1)
        W =  tf.get_variable(name="W", shape=[m, r_components], initializer=initializer)
        H =  tf.get_variable("H", [r_components, n], initializer=initializer)

        WH =tf.matmul(W, H)

        cost = tf.reduce_sum(tf.square(V - WH))
        
        if optimizer=='mu':
            """Compute Non-negative Matrix Factorization with Multiplicative Update"""
            """update H in Multiplicative Update NMF"""
            Wt = tf.transpose(W)
            H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
            H_update = H.assign(H_new)

            """update W in Multiplicative Update NMF"""
            Ht = tf.transpose(H)
            W_new = W * tf.matmul(V, Ht)/ tf.matmul(W, tf.matmul(H, Ht))
            W_update = W.assign(W_new)
        elif optimizer=='pg':
            """optimization; Projected Gradient method """
            dW, dH = tf.gradients(xs=[W, H], ys=cost)
            W_update = W.assign(W - learning_rate * dW)
            H_update = H.assign(H - learning_rate * dH)

            #Limit the updated matrices so that all elements become nonnegative
            condition_W = tf.less(W_update, 0)
            W_update = tf.where(condition_W, tf.zeros_like(W_update), W_update)
            condition_H = tf.less(H_update, 0)
            H_update = tf.where(condition_H, tf.zeros_like(H_update), H_update)

        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for idx in range(max_iter):
                H,W=sess.run([H_update,W_update], feed_dict={V:X})
                if (idx % display_step) == 0:
                    costValue = sess.run(cost,feed_dict={V:X})
                    print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue))
        return W, H

    def fit_transform(self, X, W=None, H=None):
        """Transform input data to W, H matrices which are the non-negative matrices."""
        W, H =  self.NMF(X=X, r_components = self.r_components, learning_rate=self.learning_rate, 
                    max_iter = self.max_iter, display_step = self.display_step, optimizer=self.optimizer )
        return W, H

    def inverse_transform(self, W, H):
        """Transform data back to its original space."""

        return np.matmul(W,H)


def main():
    V = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    
    model = NMF(r_components=2,max_iter=200,learning_rate=0.01,display_step=10, optimizer='mu')
    W, H = model.fit_transform(V)
    
    print(V)
    print(model.inverse_transform(W, H))

if __name__ == '__main__':
    main()