from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .base import NetworkUnit
from .base import Dataset
from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf
import random
import pickle

path = os.getcwd()

# FLAGS = tf.app.flags.FLAGS

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
num_examples = 10000
batch_size = 128
max_steps = 20
log_device_placement = False
train_dir = path + 'train_dir'
eval_dir = '/eval_dir'
checkpoint_dir = path + 'train_dir'
# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


class Evaluater:
    def __init__(self):
        '''tf.app.flags.DEFINE_integer('num_examples', 10000, """Number of examples to run.""")
        tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
        tf.app.flags.DEFINE_integer('max_steps', 20, """Number of batches to run.""")
        tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
        tf.app.flags.DEFINE_string('train_dir', path + 'train_dir', """Directory where to write event logs and checkpoint.""")
        tf.app.flags.DEFINE_string('eval_dir', '/eval_dir', """Directory where to write event logs.""")
        tf.app.flags.DEFINE_string('checkpoint_dir', path + 'train_dir', """Directory where to read model checkpoints.""")
        tf.app.flags.DEFINE_string('data_dir', path, """Path to the CIFAR-10 data directory.""")'''

        self.log = "****************"
        self.dtrain = Dataset()
        self.dvalid = Dataset()
        self.dataset = Dataset()
        self.dtrain.feature = self.dtrain.label = []
        data_dir = os.path.join(path, 'cifar-10-batches-bin')
        self.dvalid.feature, self.dvalid.label = self._load_batch(os.path.join(data_dir, 'test_batch.bin'))
        self.dataset.feature, self.dataset.label = self._load_data(data_dir)

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        var = self._variable_on_cpu(name, shape,
                                    tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _loss_averages(self, total_loss):
        """Generates moving average for all losses

        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        return loss_averages_op

    def _makeconv(self, inputs, hplist, node,cellist):
        """Generates a convolutional layer according to information in hplist

        Args:
        inputs: inputing data.
        hplist: hyperparameters for building this layer
        node: number of this cell
        Returns:
        tensor.
        """
        print('Evaluater:right now we are making conv layer, its node is %d, and the inputs is'%node,inputs,'and the node before it is ',cellist[node-1])
        with tf.variable_scope('conv' + str(node)) as scope:
            inputdim = inputs.shape[3]
            kernel = self._variable_with_weight_decay(
                'weights', shape=[hplist[2], hplist[2], inputdim, hplist[1]], stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', hplist[1], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            if hplist[3] == 'relu':
                return tf.nn.relu(bias, name=scope.name)
            elif hplist[3] == 'tanh':
                return tf.tanh(bias, name=scope.name)
            elif hplist[3] == 'sigmoid':
                return tf.sigmoid(bias, name=scope.name)
            elif hplist[3] == 'identity':
                return tf.identity(bias, name=scope.name)
            elif hplist[3] == 'leakyrelu':
                return tf.nn.leaky_relu(bias, name=scope.name)


    def _makepool(self, inputs, hplist):
        """Generates a pooling layer according to information in hplist

        Args:
            inputs: inputing data.
            hplist: hyperparameters for building this layer
        Returns:
            tensor.
        """
        if hplist[1] == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                                  strides=[1, hplist[2], hplist[2], 1], padding='SAME')
        elif hplist[1] == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                                  strides=[1, hplist[2], hplist[2], 1], padding='SAME')
        elif hplist[1] == 'max':
            return tf.reduce_mean(inputs,[1,2],keep_dims=True)

    def _makedense(self, inputs, hplist, node):
        """Generates dense layers according to information in hplist

        Args:
                   inputs: inputing data.
                   hplist: hyperparameters for building layers
                   node: number of this cell
        Returns:
                   tensor.
        """
        i = 0
        for neural_num in hplist.neural_size:
            name = 'dense' + str(node) + str(i)
            with tf.variable_scope(name) as scope:
                dim = 1
                for d in inputs.get_shape()[1:].as_list():
                    dim *= d  # Move everything into depth.
                reshape = tf.reshape(inputs, [batch_size, dim])
                weights = self._variable_with_weight_decay('weights', shape=[dim, neural_num], stddev=0.04, wd=0.004)
                biases = tf.get_variable('biases', [neural_num], initializer=tf.constant_initializer(0.1))
                if hplist.activation == 'relu':
                    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
                elif hplist.activation == 'tanh':
                    local3 = tf.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)
                elif hplist.activation == 'sigmoid':
                    local3 = tf.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)
                elif hplist.activation == 'identity':
                    local3 = tf.identity(tf.matmul(reshape, weights) + biases, name=scope.name)
            inputs = local3
            i += 1
        return inputs

    def _inference(self, images, graph_part, cellist):
        '''Method for recovering the network model provided by graph_part and cellist.
        Args:
          images: Images returned from Dataset() or inputs().
          graph_part: The topology structure of th network given by adjacency table
          cellist:

        Returns:
          Logits.'''
        print('Evaluater:starting to reconstruct the network')
        nodelen = len(graph_part)
        inputs = [0 for i in range(nodelen)]  # input list for every cell in network
        inputs[0] = images
        getinput = [False for i in range(nodelen)]  # bool list for whether this cell has already got input or not
        getinput[0] = True
        # bool list for whether this cell has already been in the queue or not
        inqueue = [False for i in range(nodelen)]
        inqueue[0] = True
        q = []
        q.append(0)

        # starting to build network through width-first searching
        while len(q) > 0:
            # making layers according to information provided by cellist
            node = q.pop(0)
            print('Evaluater:right now we are processing node %d'%node,', ',cellist[node])
            if cellist[node][0] == 'conv':  # isinstance(cellist[node], ConvolutionalCell):
                layer = self._makeconv(inputs[node], cellist[node], node,cellist)
            elif cellist[node][0] == 'pooling':  # isinstance(cellist[node], PoolingCell):
                layer = self._makepool(inputs[node], cellist[node])
            else:
                print('WRONG!!!!! Notice that you got a layer type we cant process!',cellist[node][0])
                layer=[]
            # update inputs information of the cells below this cell
            for j in graph_part[node]:
                if getinput[j]:  # if this cell already got inputs from other cells precedes it
                    # padding
                    a = int(layer.shape[1])
                    b = int(inputs[j].shape[1])
                    pad = abs(a - b)
                    if layer.shape[1] > inputs[j].shape[1]:
                        inputs[j] = tf.pad(inputs[j], [[0, 0], [0, pad], [0, pad], [0, 0]])
                    if layer.shape[1] < inputs[j].shape[1]:
                        layer = tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])
                    inputs[j] = tf.concat([inputs[j], layer], 3)
                else:
                    inputs[j] = layer
                    getinput[j] = True
                if not inqueue[j]:
                    q.append(j)
                    inqueue[j] = True

        dim = 1
        for d in inputs[nodelen - 1].get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(inputs[nodelen - 1], [batch_size, dim])
        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [dim, NUM_CLASSES], stddev=1 / float(dim), wd=0.0)
            biases = self._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)

        return softmax_linear

    def _calloss(self, logits, labels):
        """Add L2Loss to all the trainable variables.

          Add summary for for "Loss" and "Loss/avg".
          Args:
            logits: Logits from inference().
            labels: Labels, 1-D tensor of shape [batch_size]

          Returns:
            Loss tensor of type float."""
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        print(sparse_labels.shape)
        concated = tf.concat([indices, sparse_labels], 1)
        dense_labels = tf.sparse_to_dense(concated, [batch_size, NUM_CLASSES], 1.0, 0.0)

        # Calculate the average cross entropy loss across the batch.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=dense_labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
          total_loss: Total loss from loss().
        Returns:
          loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def _train_op(self, total_loss, global_step):
        """Train CIFAR-10 model.

        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.

        Args:
          total_loss: Total loss from loss().
          global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
          train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train(self, graph_part, cellist):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)
            print()
            trainlabel = tf.placeholder(tf.int32)  # .convert_to_tensor(self.dtrain.label)
            traind = tf.placeholder(tf.float32,
                                    [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])  # .convert_to_tensor(self.dtrain.feature)
            # Get images and labels.
            # logits = self._inference(images, graph_part, cellist)
            logits = self._inference(traind, graph_part, cellist)
            loss = self._calloss(logits, trainlabel)  # Calculate loss.
            num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            tf.summary.scalar('learning_rate', lr)
            train_op = self._train_op(loss, global_step)
            # Compute gradients.
            # Create a saver.
            saver = tf.train.Saver(tf.all_variables())

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(
                log_device_placement=log_device_placement))
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            for step in range(max_steps):
                start_time = time.time()
                start_index = np.random.randint(0, self.get_train_size() - batch_size)

                data = self.dtrain.feature[start_index:(start_index + batch_size)]
                label = self.dtrain.label[start_index:(start_index + batch_size)]
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={traind: data, trainlabel: label})
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    examples_per_sec = 1
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == max_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

    def evaluate(self, network=None):
        '''Train'''
        print('Evaluater:start training')
        cellist = network.cell_list[-1]
        self.train(network.graph_part, cellist)  # start training
        """Eval"""
        print('Evaluater:start evaluating')
        with tf.Graph().as_default():
            # this way of reading data is tooooooo slow!!! Need to find another way out.
            input_queue = tf.train.slice_input_producer([self.dvalid.feature, self.dvalid.label], shuffle=False)
            valid, label = tf.train.batch(input_queue, batch_size=batch_size, num_threads=16, capacity=500,
                                          allow_smaller_final_batch=False)

            logits = self._inference(valid, network.graph_part, cellist)
            top_k_op = tf.nn.in_top_k(logits, label, 1)  # Calculate predictions.self.dvalid.
            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return
                # Start the queue runners.
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                    num_iter = int(math.ceil(num_examples / batch_size))
                    true_count = 0  # Counts the number of correct predictions.
                    total_sample_count = num_iter * batch_size
                    step = 0
                    while step < num_iter and not coord.should_stop():
                        predictions = sess.run(
                            [top_k_op])  # ,feed_dict={valid:self.dvalid.feature,label:self.dvalid.label})
                        true_count += np.sum(predictions)
                        step += 1
                    precision = true_count / total_sample_count  # Compute precision.
                    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                except Exception as e:
                    coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)
        return precision

    def add_data(self, add_num=0):
        print('Evaluater: Adding data')
        if add_num:
            catag = 10
            for cat in range(catag):
                num_train_samples = self.dataset.label.shape[0]
                cata_index = [i for i in range(num_train_samples) if self.dataset.label[i] == cat]
                selected = sorted(random.sample(cata_index, int(add_num / catag)))
                if self.dtrain.feature == []:
                    self.dtrain.feature = self.dataset.feature[selected]
                    self.dtrain.label = [cat for i in range(int(add_num / catag))]
                else:
                    self.dtrain.feature = np.concatenate([self.dtrain.feature, self.dataset.feature[selected]], axis=0)
                    self.dtrain.label = np.concatenate([self.dtrain.label, [cat for i in range(int(add_num / catag))]],
                                                       axis=0)
                skip = [i for i in range(num_train_samples) if not (i in selected)]
                self.dataset.feature = self.dataset.feature[skip]
                self.dataset.label = self.dataset.label[skip]
        return 0

    def get_train_size(self):
        return len(self.dtrain.label)

    '''def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        batch_size):
        """Construct a queued batch of images and labels.

        Args:
          image: 3-D Tensor of [height, width, 3] of type.float32.
          label: 1-D Tensor of type.int32
          min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
          batch_size: Number of images per batch.

        Returns:
          images: Images. 4D tensor of [batch_size, height, width, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        num_preprocess_threads = 16
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

        return images, tf.reshape(label_batch, [batch_size])

    def _read_cifar10(self, filename_queue):
        """Reads and parses examples from CIFAR10 data files.

        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.

        Args:
          filename_queue: A queue of strings with the filenames to read from.

        Returns:
          An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
              for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
        """

        class CIFAR10Record(object):
            pass

        result = CIFAR10Record()

        # Dimensions of the images in the CIFAR-10 dataset.
        label_bytes = 1
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(
            tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                 [result.depth, result.height, result.width])
        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def _inputs(self, eval_data, data_dir, batch_size):
        """Construct input for CIFAR evaluation using the Reader ops.

        Args:
          eval_data: bool, indicating if one should use the train or eval data set.
          data_dir: Path to the CIFAR-10 data directory.
          batch_size: Number of images per batch.

        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                         for i in range(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self._read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               width, height)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(float_image, read_input.label,
                                                    min_queue_examples, batch_size)'''

    def _load_batch(self, filename):
        """ load single batch of cifar """
        bytestream = open(filename, "rb")
        buf = bytestream.read(10000 * (1 + 32 * 32 * 3))
        bytestream.close()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(10000, 1 + 32 * 32 * 3)
        data = np.float32(data)
        labels_images = np.hsplit(data, [1])
        labels = labels_images[0].reshape(10000)
        labels = np.int32(labels)
        images = labels_images[1].reshape(10000, 32, 32, 3)
        return images, labels

    def _load_data(self, ROOT):
        """ load all of cifar """
        print('Evaluater: loading data')
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d.bin' % (b,))
            X, Y = self._load_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)  # 使变成行向量
        Ytr = np.concatenate(ys)
        del X, Y
        return Xtr, Ytr


if __name__ == '__main__':
    eval = Evaluator()
    eval.add_data(1280)
    network = NetworkUnit()
    network.graph_part = [[1, 7], [2, 9], [3], [4, 5], [5], [6], [], [8], [3], [3]]
    cellist = []
    for i in range(len(network.graph_part)):
        k = random.randint(1, 2)
        # if i>7:
        #    cellist.append(cifar10.DenseCell())
        if k == 1:
            cellist.append(ConvolutionalCell())
        if k == 2:
            cellist.append(PoolingCell())
    network.cell_list = [cellist]
    eval.add_data(1280)
    e = eval.evaluate(network)
    print(e)
