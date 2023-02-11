class Autoencoder(tf.keras.Model):
    
    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 decoder_conv_filters, decoder_conv_kernel_size, decoder_conv_strides, activation, z_dim,
                 use_batch_normalization=False, dropout = None):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.z_dim = z_dim
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.dropout = dropout
        
        self.encoder, encoder_input, encoder_output = self.__create_encoder() 
        self.decoder = self.__create_decoder()
        self.model = Model(encoder_input, self.decoder(encoder_output))
        
    def __create_encoder(self):

        encoder_input = Input(shape=self.input_dim, name='encoder_input') 
        x = encoder_input
        for i in range(len(self.encoder_conv_filters)):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i],
                kernel_size = self.encoder_conv_kernel_size[i],
                strides = self.encoder_conv_strides[i],
                padding = 'same',
                activation = self.activation if isinstance(self.activation, str) else None,
                name = 'encoder_conv_' + str(i)
            )
 
            x = conv_layer(x) #(2)
    
            if inspect.isclass(self.activation) and Layer in self.activation.__bases__:
                x = self.activation()(x)
                
            if self.use_batch_normalization:
                x = BatchNormalization()(x)

            if self.dropout:
                x = Dropout(self.dropout)(x)                

        self.__shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x) #(3)
        
        encoder_output = self._create_latent_vector(encoder_input, x)
        
        return Model(encoder_input, encoder_output), encoder_input, encoder_output #(5)
        
    def _create_latent_vector(self, encoder_input, x):
        encoder_output= Dense(self.z_dim, name='encoder_output')(x) 
        return encoder_output
    
    
    def __create_decoder(self):

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input') #(1)
        
        x = Dense(np.prod(self.__shape_before_flattening))(decoder_input) #(2)
        
        
        x = Reshape(self.__shape_before_flattening)(x) #(3)
        
        for i in range(len(self.decoder_conv_filters)):
            activation = self.activation if isinstance(self.activation, str) else None
            conv_layer = Conv2DTranspose(
                filters = self.decoder_conv_filters[i],
                kernel_size = self.decoder_conv_kernel_size[i],
                strides = self.decoder_conv_strides[i],
                padding = 'same',
                activation = activation if i < len(self.decoder_conv_filters)-1 else 'sigmoid',
                name = 'decoder_conv_' + str(i)
            )
 
            x = conv_layer(x) #(4)
    
            if i < len(self.decoder_conv_filters)-1:
                if inspect.isclass(self.activation) and Layer in self.activation.__bases__:
                    x = self.activation()(x)

                if self.use_batch_normalization:
                    x = BatchNormalization()(x)

                if self.dropout:
                    x = Dropout(self.dropout)(x)                
            
            
        decoder_output = x
        
        return Model(decoder_input, decoder_output) #(6)
    
class VariationalAutoencoder(Autoencoder):
    
    def __init__(self, eta=1, alpha=0.001, loss_type='cross_entropy', **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)
        
        self.eta = eta
        self.alpha = alpha
        self.loss_type = loss_type
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
    def _create_latent_vector(self, encoder_input, x):
        
        self._mu = Dense(self.z_dim, name='mu')(x)                  #(1)                        
        self._log_var = Dense(self.z_dim, name='log_var')(x)
        
        self._encoder_mu_log_var = Model(encoder_input, (self._mu, self._log_var))    #(2)
        
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon
        
        self.encoder_output = Lambda(sampling, name='encoder_output')([self._mu, self._log_var])   #(3)

        return self.encoder_output
    
    def call(self, data):
        (z_mean, z_log_var), z = self._encoder_mu_log_var(data), self.encoder(data)
        reconstruction = self.decoder(z)
        data = data
        
        if self.loss_type == 'cross_entropy':
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
        elif self.loss_type == 'ssim':
            reconstruction_loss = 1 - tf.reduce_mean(tf.image.ssim(data, reconstruction, 2.0))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = self.eta * reconstruction_loss +  self.alpha * kl_loss
        
        # actualizamos las métricas
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='loss', aggregation='mean')
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        return reconstruction
    
    def train_step(self, data):

        with tf.GradientTape() as tape:
            (z_mean, z_log_var), z = self._encoder_mu_log_var(data), self.encoder(data)
            reconstruction = self.decoder(z)
            #data = data[0]

            if self.loss_type == 'cross_entropy':
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                )
            elif self.loss_type == 'ssim':
                data = data[0]
                reconstruction_loss = 1 - tf.reduce_mean(tf.image.ssim(data, reconstruction, 2.0))

            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = self.eta * reconstruction_loss +  self.alpha * kl_loss
            
        # obtenemos los gradientes        
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # actualizamos los pesos
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # actualizamos las métricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }