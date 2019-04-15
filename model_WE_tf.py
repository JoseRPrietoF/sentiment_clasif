import numpy as np
import tensorflow as tf
from tf_utils import CNN, FF, RNN, train_ops, prepare_text
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


class Model:

    def __init__(self,
                 X_train,
                 X_test,
                 y_train,
                 y_test,
                 fnames_train,
                 fnames_test,
                 EMBEDDING_DIM=300,
                MAX_SEQUENCE_LENGTH = None,
                 filters=[64, 128, 256, 512],
                 MODEL="CNN",
                 HIDDEN_UNITS=32,
                 NUM_LAYERS=2,
                 OPTIMIZER='adam',
                 BATCH_SIZE=64,
                 NUM_EPOCH=50,
                 n_classes=4,
                 logger=None,
                 opts=None,
                 vocab=[],
                 dir_word_embeddings='/data2/jose/word_embedding/fastext/fasttext-sbwc.vec'):

        """
        Vars
        """



        # MODEL = "RNN"
        # MODEL = "CNN"
        X_train, X_test, embedding, MAX_SEQUENCE_LENGTH = prepare_text.prepare_data(
            dir_word_embeddings, X_train, X_test, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, vocab=vocab)

        glove_weights_initializer = tf.constant_initializer(embedding)
        print(glove_weights_initializer)
        embeddings = tf.Variable(
            # tf.random_uniform([2000, EMBEDDING_DIM], -1.0, 1.0)
            embedding
        )

        print(embeddings)


        labelencoder = LabelEncoder()  # set
        y_train_ = np.array(y_train).astype(str)
        y_test_ = np.array(y_test).astype(str)
        labelencoder.fit(y_train_)
        y_train_ = labelencoder.transform(y_train_)
        y_test_ = labelencoder.transform(y_test_)
        n_values = len(np.unique(y_train_))
        # To One hot
        y_train = to_categorical(y_train_, n_values)
        y_test = to_categorical(y_test_, n_values)

        logger.info("texts_rep_train: {}".format(X_train.shape))
        logger.info("y_train: {}".format(y_train.shape))




        """""""""""""""""""""""""""""""""""
        # Tensorflow
        """""""""""""""""""""""""""""""""""

        batch_size = tf.placeholder(tf.int64, name="batch_size")
        X = tf.placeholder(tf.int32, shape=[None, MAX_SEQUENCE_LENGTH], name="X")

        print(X)
        y = tf.placeholder(tf.int64, shape=[None, n_classes], name="y")
        seq_len = tf.placeholder(tf.int32, [None], name='lengths')
        fnames_plc = tf.placeholder(tf.string, shape=[None], name="fnames_plc")
        lr = tf.placeholder(tf.float32, shape=[], name="lr")
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout")

        """
        GET THE MODEL
        """
        if MODEL == "CNN":
            logits = CNN.get_model(X, W=embeddings, is_training=is_training, filters=filters, n_classes=n_classes, logger=logger)
        elif MODEL == "RNN":
            logits = RNN.get_model(X, W=embeddings, dropout_keep_prob=dropout_keep_prob, hidden_size=HIDDEN_UNITS, n_classes=n_classes,
                                   num_layers=NUM_LAYERS, logger=logger)

        logger.info(logits)
        softmax = tf.nn.softmax(logits)

        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        logger.info("{} params to train".format(num_params))

        train_op, loss = train_ops.train_op(logits, y, learning_rate=lr, optimizer=OPTIMIZER)
        """"""
        """Test de embeddings"""

        train_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        dev_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)
        test_dataset = tf.data.Dataset.from_tensor_slices((X, y, fnames_plc)).batch(batch_size).shuffle(buffer_size=12)


        train_data = (X_train, y_train, fnames_train)

        test_data = (X_test, y_test, fnames_test)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_dataset)
        dev_init_op = iter.make_initializer(dev_dataset)
        test_init_op = iter.make_initializer(test_dataset)

        epoch_start = 0
        ## Train
        sess = tf.Session()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        for epoch in range(epoch_start, NUM_EPOCH + 1):
            sess.run(train_init_op, feed_dict={
                X: train_data[0],
                y: train_data[1],
                fnames_plc: train_data[2],
                batch_size: BATCH_SIZE,
                seq_len: [MAX_SEQUENCE_LENGTH]*BATCH_SIZE
            }
                     )

            current_batch_index = 0
            next_element = iter.get_next()
            loss_count = 0
            while True:

                try:
                    data = sess.run([next_element])
                except tf.errors.OutOfRangeError:
                    break

                current_batch_index += 1
                data = data[0]
                batch_x, batch_tgt, batch_fnames = data

                _, loss_result = sess.run([train_op, loss],
                                          feed_dict={
                                              X: batch_x,
                                              y: batch_tgt,
                                              lr: train_ops.lr_scheduler(epoch),
                                              batch_size: BATCH_SIZE,
                                              is_training: True,
                                              dropout_keep_prob: 0.2,
                                              seq_len: [MAX_SEQUENCE_LENGTH] * BATCH_SIZE

                                          })
                # print("Loss: {}".format(loss_result))
                loss_count += loss_result

            loss_count = loss_count / current_batch_index
            logger.info("Loss on epoch {} : {} - LR: {}".format(epoch, loss_count, train_ops.lr_scheduler(epoch)))
            acc = 0
            # if do_val:
            #     print("Eval")
            #     ## Eval
            #     sess.run(dev_init_op, feed_dict={
            #     # sess.run(dev_init_op, feed_dict={
            #         X: dev_data[0],
            #         y: dev_data[1],
            #         batch_size: BATCH_SIZE,
            #     }
            #              )
            #
            #     current_batch_index = 0
            #     next_element = iter.get_next()
            #
            #
            #     while True:
            #
            #         try:
            #             data = sess.run([next_element])
            #         except tf_utils.errors.OutOfRangeError:
            #             break
            #
            #         current_batch_index += 1
            #         data = data[0]
            #         batch_x, batch_tgt = data
            #
            #         results = sess.run([softmax],
            #                                   feed_dict={
            #                                       X: batch_x,
            #                                       y: batch_tgt,
            #                                       batch_size: BATCH_SIZE,
            #                                       dropout_keep_prob: 1.0
            #                                   })
            #
            #         for i in range(len(results)):
            #             acc_aux = metrics.accuracy(X=results[i], y=batch_tgt[i])
            #             acc += acc_aux
            #
            #     acc = acc / current_batch_index
            #     print("Acc Val epoch {} : {}".format(epoch, acc))
            #     print("----------")

        """
        ----------------- TEST -----------------
        """
        logger.info("\n-- TEST --\n")

        sess.run(test_init_op, feed_dict={
            X: test_data[0],
            y: test_data[1],
            fnames_plc: test_data[2],
            batch_size: BATCH_SIZE,
            seq_len: [MAX_SEQUENCE_LENGTH] * BATCH_SIZE
        }
                 )
        # sess.run(train_init_op, feed_dict={
        #     X: train_data[0],
        #     y: train_data[1],
        #     fnames_plc: train_data[2],
        #     batch_size: BATCH_SIZE,
        # }
        #          )

        current_batch_index = 0
        next_element = iter.get_next()
        pred = []
        tgt = []
        while True:

            try:
                data = sess.run([next_element])
            except tf.errors.OutOfRangeError:
                break

            current_batch_index += 1
            data = data[0]
            batch_x, batch_tgt, batch_fnames = data

            results = sess.run([softmax],
                               feed_dict={
                                   X: batch_x,
                                   y: batch_tgt,
                                   batch_size: BATCH_SIZE,
                                   dropout_keep_prob: 1.0,
                                   is_training: False,
                                   lr: train_ops.lr_scheduler(1),
                                   seq_len: [MAX_SEQUENCE_LENGTH] * BATCH_SIZE
                               })
            cs = []
            cs_tgt = []
            for i in range(len(results[0])):

                # to write
                hyp = [np.argmax(results[0][i], axis=-1)]
                hyp = labelencoder.inverse_transform(hyp)[0]  # real label   #set
                cs.append(hyp)

                tgt_epoch = [np.argmax(batch_tgt[i], axis=-1)]
                tgt_epoch = labelencoder.inverse_transform(tgt_epoch)[0]  # real label   #set
                cs_tgt.append(tgt_epoch)
            pred.extend(cs)
            tgt.extend(cs_tgt)

        cm = confusion_matrix(tgt, pred)
        acc = accuracy_score(tgt, pred)
        f1_micro = f1_score(tgt, pred, average='micro')
        f1_macro = f1_score(tgt, pred, average='macro')
        logger.info("Evaluando {}".format(MODEL))
        logger.info("Confusion matrix: \n")
        logger.info("\n"+cm)
        logger.info("ACC: {}".format(acc))
        logger.info("f1_micro: {}".format(f1_micro))
        logger.info("f1_macro: {}".format(f1_macro))
        logger.info("-------------")

        # logger.info("Writting results in output dir {}".format(opts.o))
        # [print(x) for x in classifieds_to_write]

