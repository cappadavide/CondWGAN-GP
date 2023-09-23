from tensorflow import keras
from keras import layers
import tensorflow as tf
from customLayers import UpsampleBlock, ConvBlock

class Generator(layers.Layer):
    def __init__(self,latent_dim,num_classes,**kwargs):
        super(Generator, self).__init__(name='cdwgangp-generator')
        self.dense = layers.Dense((latent_dim + num_classes) * 7 * 7, use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU(0.2)
        self.reshape = layers.Reshape((7,7,(latent_dim+num_classes)))
        self.upblock1 = UpsampleBlock(128,layers.LeakyReLU(0.2),use_bias=False)
        self.upblock2 = UpsampleBlock(64,layers.LeakyReLU(0.2),use_bias=False)
        self.outputlayer = layers.Conv2D(1,kernel_size=(3,3),strides=(1,1),activation="tanh",padding="same")
        super(Generator, self).__init__(**kwargs)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.leakyrelu(x)
        x = self.reshape(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.outputlayer(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense': self.dense,
            'batch_norm': self.batch_norm,
            'leakyrelu': self.leakyrelu,
            'reshape': self.reshape,
            'upblock1': self.upblock1,
            'upblock2': self.upblock2,
            'outputlayer': self.output

        })
        return config
    

class Critic(layers.Layer):
    def __init__(self,steps=3,**kwargs):
        super(Critic, self).__init__(name='cdwgangp-critic')
        self.steps = steps
        self.conv1 = ConvBlock(32,layers.LeakyReLU(0.2),kernel_size=(5,5),strides=(2,2),use_bias=True)
        self.conv2 = ConvBlock(64,layers.LeakyReLU(0.2),kernel_size=(5,5),strides=(2,2),use_bias=True,use_dropout=True)
        self.conv3 = ConvBlock(128,layers.LeakyReLU(0.2),kernel_size=(5,5),strides=(2,2),use_bias=True,use_dropout=True)
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.2)
        self.outputlayer = layers.Dense(1)
        super(Critic, self).__init__(**kwargs)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.outputlayer(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'steps': self.steps,
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'flatten': self.flatten,
            'dropout': self.dropout,
            'outputlayer': self.outputlayer

        })
        return config



class ConditionalWGAN_GP(keras.Model):
    def __init__(self,latent_dim,num_classes,image_size,critic_steps=3,gp_weight=10.0):
        super().__init__()
        self.gp_weight = gp_weight
        self.latent_dim = latent_dim       
        self.num_classes = num_classes 
        self.generator = Generator(latent_dim,num_classes)
        self.image_size = image_size
        self.critic = Critic(critic_steps)
    
    def compile(self,d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, image_labels):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        interpolated_labels = tf.concat([interpolated, image_labels], -1)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_labels)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated_labels, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated_labels])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self,data):
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]
        image_labels = labels[:, :, None, None]
        image_labels = tf.repeat(
            image_labels, repeats=[self.image_size * self.image_size] 
        )
        image_labels = tf.reshape(
            image_labels, (-1, self.image_size, self.image_size, self.num_classes) 
        )
        real_images_labels = tf.concat([real_images, image_labels], -1)
        for i in range(self.critic.steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_vector_labels = tf.concat([random_latent_vectors, labels], axis=1)
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_vector_labels,training=True)
                fake_images_labels = tf.concat([fake_images, image_labels], -1)
                fake_logits = self.critic(fake_images_labels, training=True)
                real_logits = self.critic(real_images_labels,training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images,image_labels)
                d_loss = d_cost + gp * self.gp_weight
            
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables)
            )
        
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, labels], axis=1)
       
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_vector_labels, training=True)
            # Get the discriminator logits for fake images
            generated_images_labels = tf.concat([generated_images, image_labels], -1)

            gen_img_logits = self.critic(generated_images_labels, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}