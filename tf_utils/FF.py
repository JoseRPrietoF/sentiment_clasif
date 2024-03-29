import tensorflow as tf

def get_model(net, dropout_keep_prob, is_training, layers=[512,256,128,64], n_classes=4, SEED=43):

    for l in layers:
        net = tf.layers.dense(net, l)
        net = tf.nn.relu(net)
        net = tf.layers.batch_normalization(net, momentum=0.9)
        net = tf.layers.dropout(net, dropout_keep_prob, training=is_training, seed=SEED)


    net = tf.layers.dense(net, n_classes)



    return net