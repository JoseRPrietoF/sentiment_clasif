import tensorflow as tf
from tensorflow.contrib import rnn


def cell(hidden_size, dropout_keep_prob, seed=None):
    """
    Builds a LSTM cell with a dropout wrapper
    :param hidden_size: Number of units in the LSTM cell
    :param dropout_keep_prob: Tensor holding the dropout keep probability
    :param seed: Optional. Random state for the dropout wrapper
    :return: LSTM cell with a dropout wrapper
    """
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                 output_keep_prob=dropout_keep_prob, seed=seed)
    return dropout_cell

def get_model(X, dropout_keep_prob, W=None, hidden_size = 32, num_layers=2 , n_classes=23, logger=None,seq_len=None):
    logger.info("CREATING THE MODEL \n")
    tf.logging.set_verbosity(tf.logging.FATAL)
    logger.info(X)

    logger.info("From Embedding")
    net = tf.nn.embedding_lookup(W, X)

    # net = tf_utils.expand_dims(X, axis=-1)  # Change the shape to [batch_size,1,max_length,output_size]
    net = tf.cast(net, tf.float32)
    logger.info("Model representation {}".format(net))
    x_len = tf.cast(tf.reduce_sum(tf.sign(X), 1), tf.int32)
    logger.info(x_len)
    # Recurrent Neural Network
    with tf.name_scope('rnn_layer'):
        fw_cells = [rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
        bw_cells = [rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
        fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob) for cell in fw_cells]
        bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob) for cell in bw_cells]
        logger.info(fw_cells)
        logger.info(bw_cells)
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, net, sequence_length=x_len, dtype=tf.float32)
        logger.info(outputs)
    #     last_output = rnn_outputs[:, -1, :]
    #     logger.info(last_output)
    #     # Final scores and predictions

        # Build LSTM cell
        # lstm_cell = cell(hidden_size, dropout_keep_prob, 43)
        # logger.info(lstm_cell)

        # outputs, _ = tf.nn.dynamic_rnn(lstm_cell, net, dtype=tf.float32, sequence_length=seq_len)
        logger.info(outputs)
        # Current shape of outputs: [batch_size, max_seq_len, hidden_size]. Reduce mean on index 1
        last_output = tf.reduce_mean(outputs, reduction_indices=[1])
        logger.info(last_output)
    with tf.name_scope("output"):

        logits = tf.layers.dense(last_output, n_classes)

    return logits

