import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
import datetime
import time
import os
import dataset
from wnet_qn import xnet
from discriminator import get_doublegan_D, DiscLossWGANGP, DoubleGAN
# print(tf.__version__)

def de_normalize(image):
    d_image = tf.cast((image + 1) * 127.5, tf.float32)
    d_image = tf.keras.applications.vgg19.preprocess_input(d_image)
    return d_image

def vgg_loss(outputs, targets):
    outputs, targets = map(de_normalize, (outputs, targets))
    perceptual_fake, perceptual_real = map(vgg, (outputs, targets))
    perceptual_loss = tf.math.reduce_mean(tf.math.square(perceptual_real - perceptual_fake))
    return perceptual_loss

@tf.function
def train_step(inputs, gt, epoch, generator, discriminator, optimizer_g, optimizer_d, gan):
    with tf.GradientTape(persistent = True) as tape:
        generator_image, res = generator(inputs, training = True)


        adv_loss = gan.loss_g(generator_image, gt)

        d_loss = gan.loss_d(generator_image, gt)

        l1_loss = tf.math.reduce_mean(tf.math.abs(generator_image-gt))
            # l2_loss = tf.math.reduce_mean(tf.math.square(generator_image-gt))

        loss_ssim = tf.math.reduce_mean(1- tf.image.ssim(generator_image, gt, max_val=2))

        perceptual_loss = vgg_loss(generator_image, gt)

        g_loss = 0.5*l1_loss + 0.1*perceptual_loss + adv_loss +0.05*loss_ssim

    psnr = tf.math.reduce_mean(tf.image.psnr(generator_image,gt,max_val=2))
    gradients_d = tape.gradient(d_loss, gan.get_trainable_variables())
    optimizer_d.apply_gradients(zip(gradients_d, gan.get_trainable_variables()))

    gradients_g = tape.gradient(g_loss, generator.trainable_variables+gan.get_trainable_variables())
    optimizer_g.apply_gradients(zip(gradients_g, generator.trainable_variables+gan.get_trainable_variables()))
    
    return g_loss, psnr, adv_loss, d_loss

@tf.function
def valid_step(inputs, groundtruth, epoch, generator):
    generator_image = generator(inputs)[0]
    l1_loss = tf.math.reduce_mean(tf.math.abs(generator_image-groundtruth))
    val_psnr = tf.image.psnr(generator_image,groundtruth,max_val=2)[0]
    return l1_loss,val_psnr

    
def train(train_dataset, val_dataset, epochs, isload = True):
    # 配置生成器
    generator = xnet()
    optimizer_g = keras.optimizers.Adam(1e-5,0,0.9)

    # 配置判别器
    discriminator = get_doublegan_D()
    optimizer_d = keras.optimizers.Adam(1e-5,0,0.9)

    # 配置GAN
    criterion_d = DiscLossWGANGP()
    double_gan = DoubleGAN(discriminator, criterion_d)

    checkpoint = tf.train.Checkpoint(
                                     generator=generator,
                                     )
    if isload:
        model_path = "./model_wholedata/last19"
        checkpoint.restore(tf.train.latest_checkpoint(model_path))
        tf.print("load model from ", model_path)
    best_manager = tf.train.CheckpointManager(checkpoint, directory='model_lr5/best', checkpoint_name='best_ckpt', max_to_keep=1)
    last_manager = tf.train.CheckpointManager(checkpoint, directory='model_lr5/last', checkpoint_name='last_ckpt', max_to_keep=1)
    best_psnr = 0

    for epoch in tf.range(epochs):
        train_ganloss = 0
        train_psnr = 0
        train_adv_g = 0
        for n, (input_image, target, mask) in train_dataset.enumerate():
            gan_loss, psnr, adv_g, adv_d = train_step(input_image, target, epoch, generator, discriminator, optimizer_g, optimizer_d, double_gan)
            train_ganloss += gan_loss
            train_psnr += psnr
            train_adv_g +=adv_g
            if epoch == 0:
                print('\repoch:{}-steps:{}  gan_loss: {:.4f} psnr: {:.4f},adv_g:{}'.format(epoch, n, gan_loss, psnr,adv_g), end = '')
        tf.print('epoch:{},ave_train_ganloss:{},ave_train_psnr:{},ave_adv_g:{}'.format(epoch,train_ganloss/float(n) , train_psnr/float(n),train_adv_g/float(n)))
        
        val_l1loss = 0
        val_psnr = 0
        for n,(input_image,target,mask) in val_dataset.enumerate():
            l1_loss, psnr = valid_step(input_image, target, epoch, generator)
            val_l1loss += l1_loss
            val_psnr += psnr
        print('epoch:{},ave_val_l1loss:{},ave_val_psnr:{}'.format(epoch,val_l1loss/float(n) ,val_psnr/float(n)))
        
        last_manager.save()
        if val_psnr/float(n) >= best_psnr:
            best_psnr = val_psnr/float(n)
            best_manager.save()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= '5'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True

    vgg_19 = keras.applications.VGG19(include_top=False)
    # vgg_19.summary()
    vgg = keras.Model(inputs=vgg_19.input, outputs=vgg_19.get_layer("block5_conv4").output) # block5_conv4
    vgg.trainable = False

    trainset = dataset.makedataset(split=1)[0]
    val_dataset =  dataset.makedataset(split=0.95)[1]
    # isload = True
    train(trainset, val_dataset, 60)
