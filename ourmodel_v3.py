from tensorflow import keras
import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras.backend as K
# import tensorflow_addons as tfa

class InstanceNormalization(tf.keras.models.Model):
    def __init__(self,epsilon=1e-5,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):  
        assert len(input_shape)==4
        shape = (input_shape[-1],)
        self.built = True
        # self.pool = tf.keras.layers.AveragePooling2D(pool_size = (input_shape[1],input_shape[2]),padding='valid')



    def call(self,inputs):
        if inputs.shape[1] == None:
            return inputs
        else:
            mean = tf.keras.layers.AveragePooling2D(pool_size=(inputs.shape[1],inputs.shape[2]))(inputs)
            # print(type(mean),mean)
            mean = tf.keras.layers.Reshape(target_shape=(1,1,inputs.shape[-1]))(mean)
            variance = tf.keras.layers.AveragePooling2D(pool_size=(inputs.shape[1],inputs.shape[2]))((inputs-mean)*(inputs-mean))*inputs.shape[1]*inputs.shape[2]
            variance =tf.keras.layers.Reshape(target_shape=(1,1,inputs.shape[-1]))(variance)
            outputs = (inputs - mean) / (tf.exp(0.5*tf.math.log(variance + self.epsilon))) 

        return outputs

def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), normlization='InstanceNormalization'):
    initializer = tf.random_normal_initializer(0., 0.02)
    #使用步长为1的卷积，保持大小不变
    x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=False)(input)    
    if normlization == 'batcnorm':
        x = keras.layers.BatchNormalization(momentum=0.0)(x,training=True)
    elif normlization == 'InstanceNormalization':
        x = InstanceNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,padding='same',kernel_initializer=initializer, use_bias=False)(x)    
    if normlization == 'batcnorm':
        x = keras.layers.BatchNormalization(momentum=0.0)(x,training=True)
    elif normlization == 'InstanceNormalization':
        x = InstanceNormalization()(x)
    out = keras.layers.Add()([input, x])#残差层
    out = keras.layers.Activation('relu')(out)
    return out

def res_block_conv(input,filters, kernel_size=(3, 3), strides=(1, 1), normlization='InstanceNormalization'):
    initializer = tf.random_normal_initializer(0., 0.02)
    # 使用步长为1的卷积，保持大小不变
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            kernel_initializer=initializer, use_bias=False)(input)
    x_res= keras.layers.Conv2D(filters=filters, kernel_size=(1,1), strides=strides, padding='same',
                            kernel_initializer=initializer, use_bias=False)(input)
    if normlization == 'batcnorm':
        x = keras.layers.BatchNormalization(momentum=0.0)(x, training=True)
    elif normlization == 'InstanceNormalization':
        x = InstanceNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            kernel_initializer=initializer, use_bias=False)(x)
    if normlization == 'batcnorm':
        x = keras.layers.BatchNormalization(momentum=0.0)(x, training=True)
    elif normlization == 'InstanceNormalization':
        x = InstanceNormalization()(x)
    out = keras.layers.Add()([x_res, x])  # 残差层
    out = keras.layers.Activation('relu')(out)
    return out

def Conv_block(input, filters, kernel_size=(3, 3), strides=(2, 2),iftanh=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    if iftanh:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=initializer,activation='tanh', use_bias=False)(input) 
    else:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=False)(input)
        #x = InstanceNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        #x = keras.layers.Dropout(0.1)(x)
    return x

def deconv_Relu(input,filters,stride=2):
    x = keras.layers.Conv2DTranspose(filters,4,stride,'same')(input)
    #x = InstanceNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.Dropout(0.1)(x)
    return x

def resnet(x,filters,num_block):
    for _ in tf.range(num_block):
        x = res_block(x,filters)
    return x


def spatial_attention(input_feature):
    kernel_size = 7
    channel = input_feature.shape[-1]
    cbam_feature = input_feature


    avg_pool = keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    #assert avg_pool.shape[-1] == 1
    max_pool = keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    #assert max_pool.shape[-1] == 1
    concat = keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    #assert concat.shape[-1] == 2
    cbam_feature = keras.layers.Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    #assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = keras.layers.Permute((3, 1, 2))(cbam_feature)
    out=keras.layers.multiply([input_feature, cbam_feature])
    return out

def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = keras.layers.Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=False,
                             bias_initializer='zeros')

    shared_layer_two = keras.layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros')

    avg_pool = keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = keras.layers.Reshape((1, 1, channel))(avg_pool)
    #assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    #assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    #assert avg_pool.shape[1:] == (1, 1, channel)



    cbam_feature = keras.layers.Activation('sigmoid')(avg_pool)

    if K.image_data_format() == "channels_first":
        cbam_feature = keras.layers.Permute((3, 1, 2))(cbam_feature)
    out=keras.layers.multiply([input_feature, cbam_feature])
    return out


def model1(x,c,num_block):
    x1 = Conv_block(x,filters=c,strides=(1,1))
    x2 = Conv_block(x1,2*c)
    x3 = Conv_block(x2,4*c)
    x4 = Conv_block(x3,8*c)
    resout = resnet(x4,8*c,num_block)
    
    y1_0 = keras.layers.concatenate([resout,x4],3)
    y1_0 = deconv_Relu(y1_0,c*4)

    y1_1 = deconv_Relu(y1_0,c*2)
    y1_2 = deconv_Relu(y1_1,c)

    y_out1 = Conv_block(y1_2,3,strides=(1,1),iftanh=True)
    return y_out1


def xnet():
    # inputs = keras.layers.Input(name='blur_image' ,shape=(None,None,3))
    # y_out1=model1(inputs,c1,res_num_block)
    # output = model2(inputs,y_out1,c2,res_num_block)
    # model = keras.models.Model(inputs = inputs,outputs = [output,y_out1], name='Discriminator')
    inputs = keras.layers.Input(name='blur_image', shape=(1024, 1024, 3))
    
    inputs1=model1(inputs,16,9)
    inputs2=tf.concat([inputs1,inputs-inputs1],3)
    
    #print(inputs.shape)
    y_conv1= Conv_block(inputs2, 32, kernel_size=(3, 3), strides=(1, 1))
    #print(y_conv1.shape)
    y_conv2=Conv_block(y_conv1,64, kernel_size=(3, 3), strides=(1, 1))
    y_conv3=Conv_block(y_conv2,128, kernel_size=(3, 3), strides=(2, 2))
    y_conv4 = Conv_block(y_conv3, 128, kernel_size=(3, 3), strides=(2, 2))
    #print(y_conv4.shape)
    y_conv2_SA=spatial_attention(y_conv2)
    y_conv3_SA = spatial_attention(y_conv3)
    y_conv4_SA = spatial_attention(y_conv4)

    y_res1=res_block(y_conv4,128)
    y_res2=res_block_conv(y_res1,256)
    y_res3=res_block(y_res2,256)
    y_res4 = Conv_block(y_res3, 512, kernel_size=(3, 3), strides=(1, 1))
    y_res2_CA = channel_attention(y_res2)
    y_res3_CA = channel_attention(y_res3)
    y_res4_CA=channel_attention(y_res4)

    y_res4_out=Conv_block(y_res4_CA, 256, kernel_size=(3, 3), strides=(1, 1))
    #print(y_res4_out.shape)
    #print(y_res3_CA.shape)
    y_res3_out=res_block_conv(tf.concat([y_res4_out, y_res3_CA],axis=3),256)
    y_res2_out=res_block_conv(tf.concat([y_res3_out, y_res2_CA],axis=3),256)
    y_res1_out=res_block_conv(y_res2_out,128)

    y_conv4_out=deconv_Relu(tf.concat([y_res1_out,y_conv4_SA],axis=3),128)
    y_conv3_out=deconv_Relu(tf.concat([y_conv4_out, y_conv3_SA],axis=3),64)
    y_conv2_out = deconv_Relu(tf.concat([y_conv3_out, y_conv2_SA], axis=3), 64,1)

    out=Conv_block(y_conv2_out,32, kernel_size=(3, 3), strides=(1, 1))
    out = keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(out)
    model = keras.models.Model(inputs=inputs, outputs=[out,inputs1], name='Discriminator')
    return model



if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # inputs = keras.layers.Input(shape=(1024,1408,3))
    model = xnet()
    model.summary()






