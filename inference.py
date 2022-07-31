import tensorflow as tf
import glob,cv2
import tensorflow.keras as keras
from wnet_qn import xnet
import numpy as np
img_list = glob.glob('/data/EBB/TestBokehFree/*.png')
model_path = 'model_wholedata/last_best_0715'
outputdir = model_path +'/test_result_fullresolution'
factor = 8


if __name__ == "__main__":
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # test.py 模型使用阶段
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    generator = xnet()
    
    # optimizer_g = keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
    checkpoint = tf.train.Checkpoint(
                                    #  optimizer_d=optimizer_d,
                                     generator=generator,
                                    #  discriminator=discriminator
                                     )         # 实例化Checkpoint，指定恢复对象为model
    checkpoint.restore(tf.train.latest_checkpoint(model_path))    # 从文件恢复模型参数
    generator.trainable = False
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    for img in img_list:
        print(img)
        out_name = os.path.basename(img)
        inputs = tf.io.read_file(img)
        blur_image = tf.io.decode_jpeg(inputs,3)
        h,w,_ = blur_image.shape
        print(factor-int(blur_image.shape[1])%factor)
        if w%factor!=0:
            paddings = tf.constant([[0,0], [0,factor-int(w)%factor],[0,0]])
            blur_image = tf.pad(blur_image,paddings,'SYMMETRIC')
        blur_image = tf.cast(blur_image, tf.float32)
        blur_image = (blur_image / 127.5) - 1.0
        blur_image = tf.expand_dims(blur_image,0)


        output =  generator(blur_image)[0]


        output = (output.numpy()+1.0)*127.5
        output = output[0,:h,:w,::-1]

        # #### cat
        # compare_path = "/data/EBB/best_22_double_gan_results/"
        # cat_image = cv2.imread(os.path.join(compare_path,out_name))
        # cat_image = cat_image[:,cat_image.shape[1]*2//3:,:]
        # output =  np.concatenate((cat_image,output),axis=1)
        # ####

        cv2.imwrite(os.path.join(outputdir,out_name),output)
