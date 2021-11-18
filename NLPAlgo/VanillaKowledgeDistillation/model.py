import tensorflow as tf
import oneflow as flow
import os


class TeacherModel:
    def __init__(self, args, model_type, X, Y):
        self.X = X
        self.Y = Y
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        # self.checkpoint_dir = args.checkpoint_dir
        # self.checkpoint_file = "bigmodel"
        self.softmax_temperature = args.temperature
        # self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.initializer = flow.random_normal_initializer(stddev=0.1)

        # Store layers weight & bias
        self.weights = {
            # 'wc1': flow.get_variable(
            #     shape=[5, 5, 1, 32], # 32个5*5的卷积核，图像通道为1
            #     dtype=flow.float,
            #     initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
            #     name="%s_%s" % (self.model_type, "wc1")
            # ),
            # 'wc2': flow.get_variable(
            #     shape=[5, 5, 32, 64],
            #     dtype=flow.float,
            #     initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
            #     name="%s_%s" % (self.model_type, "wc2")
            # ),
            'wd1': flow.get_variable(
                shape=[2 * 7 * 64, 1024],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wd1")
            ),
            'wout': flow.get_variable(
                shape=[1024, self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wout")
            )
        }

        self.biases = {
            'bc1': flow.get_variable(
                shape=[32],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bc1")
            ),
            'bc2': flow.get_variable(
                shape=[64],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bc2")
            ),
            'bd1': flow.get_variable(
                shape=[1024],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bd1")
            ),
            'bout': flow.get_variable(
                shape=[self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bout")
            )
        }

        return self.build_model()
        # self.saver = tf.train.Saver()

    # def conv2d(self, x, W, b, strides=1, name=''):
    #     # Conv2D wrapper, with bias and relu activation
    #     with flow.scope.namespace("%sconv2d" % (self.model_type)):
    #         x = flow.layers.conv2d(x, filters=)
    #         x = flow.layers.conv2d(x, W, strides=[1, strides, strides, 1], data_format="NCHW",
    #                                kernel_initializer=self.initializer, padding='SAME', name=name)
    #         x = flow.nn.bias_add(x, b)
    #         return flow.nn.relu(x)

    def conv2d(self, x, filters: int, b, kernel_size: int, strides=1, name=''):
        # Conv2D wrapper, with bias and relu activation
        with flow.scope.namespace("%sconv2d" % (self.model_type)):
            x = flow.layers.conv2d(
                x,
                filters=filters,
                kernel_size=kernel_size,
                strides=[strides, strides],
                data_format="NCHW",
                kernel_initializer=self.initializer,
                padding='SAME',
                name="{}-{}".format(self.model_type, name)
            )
            x = flow.nn.bias_add(x, b)
            return flow.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        with flow.scope.namespace("%smaxpool2d" % (self.model_type)):
            return flow.nn.max_pool2d(x, ksize=[k, k], strides=[k, k],
                                  padding='SAME')

    # Create model
    def build_model(self):
        # self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        # self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))
        # self.keep_prob = tf.placeholder(tf.float32,
        #                                 name="%s_%s" % (self.model_type, "dropoutprob"))  # dropout (keep probability)

        # self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemp"))
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        with flow.scope.namespace("%sinputreshape" % (self.model_type)):
            x = flow.reshape(self.X, shape=[-1, 28, 28, 1])

        # Convolution Layer
        with flow.scope.namespace("%sconvmaxpool" % (self.model_type)):
            # print('x.shape=', x.shape)
            conv1 = self.conv2d(x, 32, self.biases['bc1'], 5, name='conv1') # 第一层卷积，32个卷积核，维度为5*5
            # Max Pooling (down-sampling)
            conv1 = self.maxpool2d(conv1, k=2)

            # Convolution Layer
            conv2 = self.conv2d(x, 64, self.biases['bc2'], 5, name='conv2') # 第一层卷积，64个卷积核，维度为5*5
            # Max Pooling (down-sampling)
            conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        with flow.scope.namespace("%sfclayer" % (self.model_type)):
            # print("conv2.shape=", conv2.shape) # [100, 64, 14, 1]
            fc1 = flow.reshape(conv2, [-1, self.weights['wd1'].shape[0]]) # 2 * 7 * 64 表示隐状态维度，与

            # print("fc1.shape=", fc1.shape) # [100, 896]
            # print("self.weights.shape=", self.weights['wd1'].shape) # [896, 1024]
            fc1 = flow.nn.bias_add(flow.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = flow.nn.relu(fc1)
            # Apply Dropout
            fc1 = flow.nn.dropout(fc1, self.dropoutprob)

            # Output, class prediction
            # print("fc1.shape=", fc1.shape)
            # print("self.weights['out'].shape=", self.weights['wout'].shape)
            # print("self.biases['out'].shape=", self.biases['bout'].shape)
            self.logits = flow.nn.bias_add(flow.matmul(fc1, self.weights['wout']), self.biases['bout']) / self.softmax_temperature

        with flow.scope.namespace("%sprediction" % (self.model_type)):
            self.prediction = flow.nn.softmax(self.logits)
        #     # Evaluate model
        #     correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with flow.scope.namespace("%soptimization" % (self.model_type)):
            # Define loss and optimizer
            # print('logits.shape=', self.logits.shape)
            # print('Y.shape=', self.Y.shape)
            self.loss = flow.math.reduce_mean(flow.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))

    def get_res(self):
        return self.loss, self.logits, self.prediction



class StudentModel:
    def __init__(self, args, model_type, X, Y, soft_Y = None, flag: bool = False):
        self.X = X
        self.Y = Y # 真实标签
        self.soft_Y = soft_Y # teacher模型获得的概率分布
        self.flag = flag # 用于判断是否添加soft_Y的loss
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.n_hidden_1 = 256  # 1st layer number of neurons
        self.n_hidden_2 = 256  # 2nd layer number of neurons
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.softmax_temperature = args.temperature
        # self.checkpoint_dir = args.checkpoint_dir
        # self.checkpoint_file = "smallmodel"
        # self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        # self.max_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + "max")
        # self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.initializer = flow.random_normal_initializer(stddev=0.1)

        # Store layers weight & bias
        self.weights = {
            'h1': flow.get_variable(
                shape=[self.num_input, self.n_hidden_1],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "h1")
            ),
            'h2': flow.get_variable(
                shape=[self.n_hidden_1, self.n_hidden_2],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "h2")
            ),
            'wout': flow.get_variable(
                shape=[self.n_hidden_2, self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wout")
            ),
            'wlinear': flow.get_variable(
                shape=[self.num_input, self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wlinear")
            )
        }

        self.biases = {
            'b1': flow.get_variable(
                shape=[self.n_hidden_1],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "b1")
            ),
            'b2': flow.get_variable(
                shape=[self.n_hidden_2],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "b2")
            ),
            'bout': flow.get_variable(
                shape=[self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bout")
            ),
            'blinear': flow.get_variable(
                shape=[self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "blinear")
            )
        }

        self.build_model()

        # self.saver = tf.train.Saver()

    # Create model
    def build_model(self):
        # self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        # self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))

        # self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        # self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "softy"))
        # self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemperature"))

        with flow.scope.namespace("%sfclayer" % (self.model_type)):
            # Hidden fully connected layer with 256 neurons
            # layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
            # print('self.X.shape=', self.X.shape)
            # print('self.weights["h1"].shape=', self.weights['h1'].shape)
            # print('self.biases["b1"].shape=', self.biases['b1'].shape)
            x = flow.reshape(self.X, shape=[-1, self.X.shape[-1] * self.X.shape[-2]])
            layer_1 = flow.nn.bias_add(flow.matmul(x, self.weights['h1']), self.biases['b1'])
            # # Hidden fully connected layer with 256 neurons
            # layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            layer_2 = flow.nn.bias_add(flow.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            # # Output fully connected layer with a neuron for each class
            # logits = (tf.matmul(layer_2, self.weights['out']) + self.biases['out'])
            self.logits = flow.nn.bias_add(flow.matmul(layer_2, self.weights['wout']), self.biases['bout'])
            # logits = flow.nn.bias_add(flow.matmul(self.X, self.weights['linear']), self.biases['linear'])
            # logits = flow.nn.bias_add(flow.matmul(layer_2, self.weights['wlinear']), self.biases['blinear'])

        with flow.scope.namespace("%sprediction" % (self.model_type)):
            self.prediction = flow.nn.softmax(self.logits)

        #     self.correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with flow.scope.namespace("%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_standard = flow.math.reduce_mean(flow.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))

            self.total_loss = self.loss_standard

            self.loss_soft = 0.0

            if self.flag:
                # print("self.logits.shape=", self.logits.shape) # [100, 10]
                # print("self.soft_Y.shape=", self.soft_Y.shape) # [100, 10]
                self.loss_soft = flow.math.reduce_mean(flow.nn.softmax_cross_entropy_with_logits(
                                            logits=self.logits / self.softmax_temperature, labels=self.soft_Y))

            self.total_loss += self.softmax_temperature * self.softmax_temperature * self.loss_soft


    def get_res(self):
        return self.total_loss, self.logits, self.prediction