import tensorflow as tf


def train_op(X,y, learning_rate=0.001, optimizer='adam'):
    y = tf.cast(y, dtype=tf.float32)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0.0001

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=X)
    # loss = tf.reduce_sum(loss) + lossL2
    loss = tf.reduce_sum(loss)
    if optimizer == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate,)
    elif optimizer == 'rms' or optimizer == 'rmsprop':
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    return optim.minimize(loss), loss


def lr_scheduler(epoch):
    if epoch < 5:
        return 0.002
    if epoch < 300:
        return 0.001

    return 0.0001




