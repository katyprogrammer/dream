import tensorflow as tf

class Seq2SeqChatbot(object):
    def __init__(self, in_sen_len, out_sen_len, vocab_size, embed_dim, hidden_dim, drop_rate, 
                 depth, max_gradient_norm, learning_rate, softmaxSamples, test, train_embed, reuse=False):
        
        print("Model creation start...")

        self.in_sen_len = in_sen_len
        self.out_sen_len = out_sen_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate
        self.depth = depth
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.softmaxSamples = softmaxSamples
        self.test = test
        self.train_embed = train_embed
        self.dtype = tf.float32
        self.reuse = reuse

        # Construct the graphs
        self.build_graph()
        
        print("Model creation end...")
    
    def build_graph(self):
        
        # define model input (placeholders)
        with tf.name_scope('placeholder_encoder'):
            # Batch size * sequence length * input dim
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None,], name='encoder_inputs') for _ in range(self.in_sen_len)]

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None,], name='decoder_inputs') for _ in range(self.out_sen_len)]
            self.decoderTargets = [tf.placeholder(tf.int32,   [None,], name='targets')        for _ in range(self.out_sen_len)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None,], name='weights')        for _ in range(self.out_sen_len)]
        
        # Creation of the rnn cell
        cells = list()
        for _ in range(self.depth):
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True, reuse=self.reuse)
            encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=1.0-self.drop_rate)
            cells.append(encoDecoCell)
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        
        # a single layer to map hidden size to vacab size
        # like time_distributed_dense after RNN in Keras

        with tf.variable_scope('weights_projection'):
            W = tf.get_variable("proj_W", (self.hidden_dim, self.vocab_size), dtype=self.dtype)
            b = tf.get_variable("proj_b", self.vocab_size, dtype=self.dtype)
        output_projection = (W, b)
        
        # do negtive sampling on loss function, for speed up
        def sampledSoftmax(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            localWt     = tf.cast(tf.transpose(output_projection[0]), tf.float32)
            localB      = tf.cast(output_projection[1],               tf.float32)
            localInputs = tf.cast(inputs,                             tf.float32)

            return tf.cast(
                tf.nn.sampled_softmax_loss(
                    localWt,
                    localB,
                    labels,
                    localInputs,
                    self.softmaxSamples,
                    self.vocab_size),
                self.dtype)
        
        # Define the network
        # Here we use attention embedding model
        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            self.encoderInputs,
            self.decoderInputs,
            encoDecoCell,
            self.vocab_size,
            self.vocab_size,
            embedding_size=self.embed_dim,
            output_projection=output_projection,
            feed_previous=self.test                    # input of decoder is from previous output or from decoder input
        )
        
        # For testing only
        if self.test:
            # decoderOutputs are hidden_size
            self.outputs = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in decoderOutputs]
        
        # For training only
        else:
            # define the loss function
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.vocab_size,
                softmax_loss_function=sampledSoftmax
            )
            # tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost
            
            # get all trainable variables
            var_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            
            if not self.train_embed:
                with tf.variable_scope("embedding_attention_seq2seq/rnn/embedding_wrapper", reuse=True):
                    em_in = tf.get_variable("embedding")
                with tf.variable_scope("embedding_attention_seq2seq/embedding_attention_decoder", reuse=True):
                    em_out = tf.get_variable("embedding")
                var_train.remove(em_in)
                var_train.remove(em_out)
            
            # initialize the optimizer
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )# .minimize(self.loss, var_list=var_train)
            
            # do gradient clipping
            gradients = tf.gradients(self.loss, var_train)
            gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.norm = tf.global_norm(gradients)
            self.optim = self.optim.apply_gradients(zip(gradients, var_train))
        
    def step(self, session, batch, isValid):

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.test:  # Training
            for i in range(self.in_sen_len):
                feedDict[self.encoderInputs[i]]  = batch[0][i]
            for i in range(self.out_sen_len):
                feedDict[self.decoderInputs[i]]  = batch[1][i]
                feedDict[self.decoderTargets[i]] = batch[2][i]
                feedDict[self.decoderWeights[i]] = batch[3][i]

            if not isValid:
                ops = (self.optim, self.loss, self.norm)
            else:  # Validation
                ops = (self.loss,)
                
        else:  # Testing (batchSize == 1)
            for i in range(self.in_sen_len):
                feedDict[self.encoderInputs[i]]  = [batch[0][i]]
            feedDict[self.decoderInputs[0]]  = [2]                      # 2 is '<beg>' token

            ops = (self.outputs,)
        
        outputs = session.run(ops, feedDict)
        
        if not self.test:  # Training
            if not isValid:
                return outputs[1], outputs[2]   # loss, norm
            else:  # Validation
                return outputs[0]               # loss
        else:  # Testing (batchSize == 1)
            return outputs[0]                   # outputs.
    
    def load_embedding(self, session, embedding):
        # Fetch embedding variables from model
        with tf.variable_scope("embedding_attention_seq2seq/rnn/embedding_wrapper", reuse=True):
            em_in = tf.get_variable("embedding")
        with tf.variable_scope("embedding_attention_seq2seq/embedding_attention_decoder", reuse=True):
            em_out = tf.get_variable("embedding")

        # Initialize input and output embeddings
        session.run(em_in.assign(embedding))
        session.run(em_out.assign(embedding))
        
        # Disable training for embeddings
        tf.stop_gradient(em_in)
        tf.stop_gradient(em_out)
        
