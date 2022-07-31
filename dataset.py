import tensorflow as tf
import os
import numpy as np
# seed = tf.random.set_seed(1000)

def load_img(path_A, path_B,path_mask):
    blur_image = tf.io.read_file(path_A)
    blur_image = tf.io.decode_jpeg(blur_image,3)
    sharp_image = tf.io.read_file(path_B)
    sharp_image = tf.io.decode_jpeg(sharp_image,3)

    mask_image = tf.io.read_file(path_mask)
    mask_image = tf.io.decode_jpeg(mask_image,1)
    
    blur_image = tf.cast(blur_image, tf.float32)
    sharp_image = tf.cast(sharp_image, tf.float32)
    mask_image = tf.cast(mask_image, tf.float32)

    
    return blur_image, sharp_image,mask_image

### 暂时用不到
def resize_img(blur_image, sharp_image, input_shape):
    blur_image = tf.image.resize(blur_image, [input_shape[0], input_shape[1]],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    sharp_image = tf.image.resize(sharp_image, [input_shape[0], input_shape[1]],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return blur_image, sharp_image


## crop的代码
def random_crop(blur_image, sharp_image,mask_image, input_shape):
    seed = np.random.randint(0,1000)    
    blur_image = tf.image.random_crop(blur_image,size=[input_shape[0], input_shape[1],3],seed=seed)
    sharp_image = tf.image.random_crop(sharp_image,size=[input_shape[0], input_shape[1],3],seed=seed)
    mask_image = tf.image.random_crop(mask_image,size=[input_shape[0], input_shape[1],1],seed=seed)
    return blur_image, sharp_image, mask_image


# def center_crop(blur_image, sharp_image,mask_image, input_shape):    
#     blur_image = tf.image.central_crop(blur_image,size=[ input_shape[0], input_shape[1],3],seed=seed)
#     sharp_image = tf.image.central_crop(sharp_image,size=[input_shape[0], input_shape[1],3],seed=seed)
#     mask_image = tf.image.central_crop(mask_image,size=[input_shape[0], input_shape[1],1],seed=seed)
#     return blur_image, sharp_image ,mask_image

# normalizing the images to [-1, 1]
def normalize(blur_image, sharp_image):
    blur_image = (blur_image / 127.5) - 1 
    sharp_image = (sharp_image / 127.5) - 1
    
    return blur_image, sharp_image

# normalizing the images to [-1, 1]
def normalize_single(image):
    image = (image / 127.5) - 1 
    
    return image

@tf.function()
def random_jitter(blur_image, sharp_image,mask_image, input_shape): 
    # random_crop to input_shape
    blur_image, sharp_image ,mask_image = random_crop(blur_image, sharp_image,mask_image, input_shape)

    if tf.random.uniform(()) > 0.5:
        blur_image = tf.image.flip_left_right(blur_image)
        sharp_image = tf.image.flip_left_right(sharp_image)
        mask_image = tf.image.flip_left_right(mask_image)
    return blur_image, sharp_image ,mask_image

@tf.function()
def crop_val_nojitter(blur_image, sharp_image,mask_image, input_shape): 
    # random_crop to input_shape
    blur_image, sharp_image ,mask_image = random_crop(blur_image, sharp_image,mask_image, input_shape)
    return blur_image, sharp_image ,mask_image

def load_img_train(path_A, path_B, path_mask,input_shape=(1024, 1408, 3),iftrain=True):
    # print(path_A)
    blur_image, sharp_image ,mask_image = load_img(path_A, path_B,path_mask)
    blur_image, sharp_image ,mask_image = random_jitter(blur_image, sharp_image, mask_image, input_shape )
    blur_image, sharp_image = normalize(blur_image, sharp_image)
    return blur_image, sharp_image ,mask_image/255.

def load_img_val(path_A, path_B, path_mask,input_shape=(1024, 1408, 3),iftrain=True):
    blur_image, sharp_image ,mask_image = load_img(path_A, path_B,path_mask)
    blur_image, sharp_image ,mask_image = crop_val_nojitter(blur_image, sharp_image, mask_image, input_shape )
    blur_image, sharp_image = normalize(blur_image, sharp_image)
    return blur_image, sharp_image ,mask_image/255.

def get_list(root,img_list,split=1.0):
    train_data_dic = {}
    val_data_dic = {}
    pathlist_bokeh = []
    pathlist_original = []
    pathlist_mask = []
    bokeh = os.path.join(root,'bokeh')
    mask = os.path.join(root,'defocus_binary_05_005298')
    original = os.path.join(root,'original')
    with open(img_list,'r') as f:
        for a in f:
            pathlist_bokeh.append(os.path.join(bokeh, a.strip('\n')))
            pathlist_original.append(os.path.join(original,a.strip('\n')))
            pathlist_mask.append(os.path.join(mask,a.strip('\n').split('.')[0]+'.png'))
    if split==1.0:
        print('nums of train data: ', len(pathlist_bokeh))
        train_data_dic['pathlist_bokeh'] = pathlist_bokeh
        train_data_dic['pathlist_original'] = pathlist_original
        train_data_dic['pathlist_mask'] = pathlist_mask
        return train_data_dic,val_data_dic
    else:
        len_train = int(len(pathlist_bokeh)*split)
        print('nums of train data: ', len_train)
        print('nums of val data: ', len(pathlist_bokeh)-len_train)
        train_data_dic['pathlist_bokeh'] = pathlist_bokeh[:len_train]
        train_data_dic['pathlist_original'] = pathlist_original[:len_train]
        train_data_dic['pathlist_mask'] = pathlist_mask[:len_train]

        val_data_dic['pathlist_bokeh'] = pathlist_bokeh[len_train:]
        val_data_dic['pathlist_original'] = pathlist_original[len_train:]
        val_data_dic['pathlist_mask'] = pathlist_mask[len_train:]   
        return train_data_dic,val_data_dic     


def makedataset(root='/data/EBB/new_training_6_5/',img_list="/data/EBB/new_training_6_5/new_list.txt",split=1.0):
    train_data_dic,val_data_dic = get_list(root,img_list,split)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_dic['pathlist_original'],train_data_dic['pathlist_bokeh'],train_data_dic['pathlist_mask']))
    train_dataset = train_dataset.map(load_img_train,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(1)   # 某个地方的数据小于1536?

    if split<1.0:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data_dic['pathlist_original'],val_data_dic['pathlist_bokeh'],val_data_dic['pathlist_mask']))
        val_dataset = val_dataset.map(load_img_val,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(1)  
    else:
        val_dataset=None
    return train_dataset,val_dataset

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import numpy as np
    import  cv2
    trainset = makedataset(split=0.9)[0]
    for inx,data in trainset.enumerate():
        print(data[0].shape)
        # data = trainset.take(1)
        # print(data)
        # input = (data[0].numpy()+1.0)*127.5
        # output = (data[1].numpy()+1.0)*127.5
        # print(inx,input.shape,output.shape)
        # # break/
        # output = np.concatenate([input,output],1)
        # # cv2.imwrite('1.png',input[...,::-1])
        # cv2.imwrite('2.png',output[...,::-1])
        # break