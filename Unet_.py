import tensorflow as tf
import numpy as np
import shutil  # high level file manipulation tool
import logging, os
import Layers

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def _get_image_summary(input_tensor):
    '''
    here extract image summary from 4D tensor
    '''
    V = tf.slice(input_tensor, (0, 0, 0, 0), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    w = tf.shape(input_tensor)[1]
    h = tf.shape(input_tensor)[2]
    V = tf.reshape(V, (w, h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, w, h, 1))
    return V

def error_rate(prediction, label):
    '''calculate the error rate based on prediction and label'''
    pred_map = np.argmax(prediction, 3)
    label_map = np.argmax(label, 3)
    pixel_volume = prediction.shape[0] * prediction.shape[1] * prediction.shape[2]
    return 100.0 - 100.0 * (np.sum(pred_map == label_map) / pixel_volume)


def create_unet(input, keep_prob, channels, n_class, layers=5, feature_root=32, filter_size=3, pool_size=2,
                summary=True):
    '''
    u_net is created here.
    :param input: input batch (or placeholder)
    :param keep_prob: used in drop-out layer
    :return: output tensor,
    '''
    logging.info('U-net details: ')
    logging.info('encoding depth: {depth}'.format(
        depth=layers
    ))

    # some preprocessing
    with tf.name_scope('reshape_beforehand'):
        nx = tf.shape(input)[1]
        ny = tf.shape(input)[2]
        input_image = tf.reshape(input, [-1, nx, ny, channels])
        in_node = input_image
        batch_size = tf.shape(input_image)[0]

    in_size = 572
    size = in_size
    dw_maps = []
    image_summary = []  # list of summary images
    variables = []  # variables in conv networks, used for regularization
    feature = feature_root

    # path down
    for layer in range(layers):
        with tf.name_scope('down_conv_{:0>2d}'.format(layer)):
            feature *= 2
            if layer == 0:
                w1 = Layers.weight_variable(shape=[filter_size, filter_size, channels, feature],
                                            name='w1')
            else:
                w1 = Layers.weight_variable(shape=[filter_size, filter_size, feature // 2, feature],
                                            name='w1')
            b1 = Layers.bias_variable(shape=[feature], name='b1')
            w2 = Layers.weight_variable(shape=[filter_size, filter_size, feature, feature],
                                        name='w2')
            b2 = Layers.bias_variable(shape=[feature], name='b2')

            # two concatenated conv
            conv1 = Layers.conv2d(in_node, w1, b1, keep_prob)
            relu_conv1 = tf.nn.relu(conv1)
            conv2 = Layers.conv2d(relu_conv1, w2, b2, keep_prob)
            relu_conv2 = tf.nn.relu(conv2)

            dw_maps.append(relu_conv2)
            image_summary.append(conv2)
            variables.append(w1)
            variables.append(b1)
            variables.append(w2)
            variables.append(b2)

            size -= 4
            in_node = relu_conv2
            if layer < layers - 1:
                # max-pooling
                pool = Layers.max_pool(relu_conv2, pool_size)
                in_node = pool
                size /= 2

    # up path
    for layer in range(-1, -layers, -1):
        with tf.name_scope('up_conv_{:0>2d}'.format(abs(layer))):
            feature = feature // 2
            size *= 2
            # up conv layer
            wd = Layers.weight_variable(shape=[pool_size, pool_size, feature, feature * 2],
                                        name='wd')
            bd = Layers.bias_variable(shape=[feature], name='bd')
            trans_conv = Layers.transposed_conv2d(input=in_node,
                                                  weight=wd,
                                                  bias=bd,
                                                  stride=pool_size)
            # crop and concatenation
            crop_concat = Layers.crop_and_concat(dw_maps.pop(), trans_conv)
            # conv1
            w1 = Layers.weight_variable(shape=[filter_size, filter_size, feature * 2, feature],
                                        name='w1')
            b1 = Layers.bias_variable(shape=[feature], name='b1')
            conv1 = Layers.conv2d(input=crop_concat,
                                  weight=w1,
                                  bias=b1,
                                  prob=keep_prob)
            relu_conv1 = tf.nn.relu(conv1)
            # conv2
            w2 = Layers.weight_variable(shape=[filter_size, filter_size, feature, feature],
                                        name='w2')
            b2 = Layers.bias_variable(shape=[feature], name='b2')
            conv2 = Layers.conv2d(input=relu_conv1,
                                  weight=w2,
                                  bias=b2,
                                  prob=keep_prob)

            image_summary.append(conv2)
            variables.append(wd)
            variables.append(bd)
            variables.append(w1)

            variables.append(b1)
            variables.append(w2)
            variables.append(b2)

            relu_conv2 = tf.nn.relu(conv2)
            size -= 4
            in_node = relu_conv2

    # 1*1 conv to output segmentation map
    with tf.name_scope('output_map'):
        wo = Layers.weight_variable(shape=[1, 1, feature, n_class], name='wo')
        bo = Layers.bias_variable([2], name='bo')
        output = Layers.conv2d(input=in_node,
                               weight=wo,
                               bias=bo,
                               prob=1.0)
        output = tf.nn.relu(output)

    if summary:
        # generate the summary of images
        with tf.name_scope('summarites'):
            # more summaries possible here
            for i, conv in enumerate(image_summary):
                tf.summary.image('summary_conv_{:0>2d}'.format(i), _get_image_summary(conv))

    return output, variables


class unet:
    def __init__(self, channels=3, n_class=2):
        '''
        An initial implementation of unet implementation:
        default setting:
            cost: cross_entropy
            summary: true
            regularizer : true
        '''
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = True

        # placeholder for training:
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, None, None, channels],
                                name='x')
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None, None, None, self.n_class],
                                name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        name='dropout_prob')

        logits, self.variables = create_unet(input=self.x,
                                             keep_prob=self.keep_prob,
                                             channels=channels,
                                             n_class=self.n_class)

        self.cost = self._get_cost(logits)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        with tf.name_scope('cross_entropy'):
            self.cross_entropy = Layers.cross_entropy(label=tf.reshape(self.y, [-1, self.n_class]),
                                                      output_map=tf.reshape(Layers.pixelwise_softmax(logits),
                                                                            [-1, self.n_class]))

        with tf.name_scope('results'):
            self.predictor = Layers.pixelwise_softmax(logits)
            self.correct_pred = tf.equal(tf.argmax(self.predictor, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits):
        '''
        cost function of unet model,
        currently only cross_entropy is supported
        regularizer is used by default
        '''

        with tf.name_scope('cost'):
            flat_logits = tf.reshape(logits, [-1, self.n_class])
            flat_labels = tf.reshape(self.y, [-1, self.n_class])

            # only cross_entropy is supported now
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                             labels=flat_labels))
            # regularizer added
            regularizers = sum([tf.nn.l2_loss(i) for i in self.variables])
            loss += regularizers * 1.0

        return loss

    def predict(self, model_path, x_test):
        '''
        use existing model to generate the prediction for the given data

        existing checkpoint is required
        x_test: [n, nx, ny, channels]
        return: prediction: [n, nx', ny', labels](input size and output size are different)
        '''

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # variables initialization
            sess.run(init)

            # restore the model from existing model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predictor, feed_dict={
                self.x: x_test,
                self.y: y_dummy,
                self.keep_prob: 1.0
            })

        return prediction

    def save(self, sess, model_path):
        '''
        save current session to a checkpoint

        :param sess: current session instance
        :param model_path: rt
        :return: None
        '''

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        '''
        restore a session from a checkpoint

        :param sess: current session
        :param model_path:
        :return:
        '''

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("model restored from: {}".format(model_path))


class Trainer:
    '''
    used to train unet
    '''

    def __init__(self, net, batch_size=1, verifi_batch_size=4, norm_grads=False, optimizer='momentum'):
        self.net = net
        self.batch_size = batch_size
        self.verificaiton_batch_size = verifi_batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == 'momentum':
            learning_rate = 0.2
            decay_rate = 0.95
            momentum = 0.2

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=momentum).minimize(self.net.cost,
                                                                               global_step=global_step)

        elif self.optimizer == "adam":
            # learning rate should modify according to momentum training
            learning_rate = 0.01
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node).minimize(self.net.cost,
                                                                                                       global_step=global_step)
        else:
            optimizer = None
            logging.warning('Wrong Optimizer is called')
        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name='global_step')

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name='norm_gradients')

        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        # summary loss and accuracy
        tf.summary.scalar('loss', self.net.loss)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters=training_iters,
                                             global_step=global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # allocate prediction and output path
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1,
              restore=False, write_graph=False, prediction_path='prediction'):
        '''

        :param data_provider:
        :param output_path:
        :param training_iters:
        :param epochs:
        :param dropout:
        :param display_step:
        :param restore:
        :param write_graph:
        :param prediction_path:
        :return:
        '''

        save_path = os.path.join(output_path, 'model.ckpt')

        init = self._initialize(training_iters, output_path, restore, prediction_path)

        with tf.InteractiveSession() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, 'graph.pb', False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider(self.verificaiton_batch_size)
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")\

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start training")

            avg_gradients = None

            # training loop
            for epoch in range(epochs):
                total_loss = 0
                for step in range(epoch * training_iters, (epoch + 1) * training_iters):
                    batch_x, batch_y = data_provider(self.batch_size)

                    # backpropagation
                    # batch_y size already resized
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict={
                            self.net.x : batch_x,
                            self.net.y : batch_y,
                            self.net.keep_prob: dropout
                        }
                    )

                    # if self.net.summaries and self.norm_grads:
                        # avg_gradients = _update_avg_gradients(avg_gradients, gradient
                        # norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        # self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                    batch_y)

                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, 'epoch_{}'.format(epoch))

                save_path = self.net.save(sess, save_path)

            logging.info('training finished')

            return save_path

    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predictor, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.0
                                                            })
        pred_shape = prediction.shape

        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                  self.net.y: batch_y,
                                                  self.net.keep_prob: 1.0})

        logging.info("Verification error={:.1f}%, loss= {:.4f}".format(error_rate(prediction, batch_y), loss))

        # img = combine_img_prediction(batch_x, batch_y, prediction)
        # save img

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info('Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}'.format(epoch,
                                                                                     (total_loss/training_iters),
                                                                                     lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # calculate batch loss and accuracy
        summary_str, loss, acc, prediction = sess.run([self.summary_op,
                                                       self.net.cost,
                                                       self.net.accuracy,
                                                       self.net.predictor],
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y,
                                                                 self.net.keep_prob: 1.0})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info('Iter {:}, Minibatch Loss= {:.4f}, Training accuracy= {:.4f}, Minibatch error= {:.1f}%'.format(step,
                                                                                                                    loss,
                                                                                                                    acc,
                                                                                                                    error_rate(prediction, batch_y)))
        









