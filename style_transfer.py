import numpy as np
from PIL import Image
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 

def load_img(path_to_img):
    
    max_dim = 512
    
    img = Image.open(path_to_img)
    
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    
    img = kp_image.img_to_array(img)    
    
    img = np.expand_dims(img, axis=0)
    
    return img

def preprocess_img(img):    
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def depreprocess_img(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)        
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def build_model(style_layers, content_layers):    
    
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', pooling = 'avg')
    vgg.trainable = False
    
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
   
    return models.Model(vgg.input, model_outputs)

def get_content_loss(content_feature, output_content_feature):

    return tf.reduce_mean(tf.square(output_content_feature - content_feature))

def gram_matrix(input_tensor):
    
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    
    return gram / tf.cast(n, tf.float32)

def get_style_loss(style_feature, output_style_feature):    
    
    style_gram = gram_matrix(style_feature)    
    output_style_gram = gram_matrix(output_style_feature)
    
    return tf.reduce_mean(tf.square(style_gram - output_style_gram))

def get_feature_representations(model, content_path, style_path, num_style_layers):  
   
    content_image = load_img(content_path)
    style_image = load_img(style_path)    
   
    content_image = preprocess_img(content_image)
    style_image = preprocess_img(style_image)
    
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

def compute_loss(model, init_image, style_features, content_features, alpha, beta):  
    
    model_outputs = model(init_image)

    style_output_features = model_outputs[:len(style_features)]
    content_output_features = model_outputs[len(style_features):]
    
    style_loss = 0
    content_loss = 0
       
    for style_feature, output_style_feature in zip(style_features, style_output_features):
        style_loss += get_style_loss(style_feature, output_style_feature[0]) / len(style_features)
    
    for content_feature, output_content_feature in zip(content_features, content_output_features):
        content_loss += get_content_loss(content_feature, output_content_feature[0]) / len(content_features)

    style_loss *= beta    
    content_loss *= alpha
    
    loss = style_loss + content_loss 
    
    return loss

def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        loss = compute_loss(**cfg)    
    
    return tape.gradient(loss, cfg['init_image']), loss

def run_style_transfer(content_path, 
                       style_path,
                       style_layers,
                       content_layers,        
                       print_iterations,
                       alpha, 
                       beta,
                       beta1,
                       beta2,
                       epsilon,
                       pooling,
                       learning_rate,
                       iterations): 
  
    tf.enable_eager_execution()
    
    #Build the model for neural style transfer
    model = build_model(style_layers, content_layers)    
    
    for layer in model.layers:
        layer.trainable = False  

    #Get feature maps for content and style images
    style_features, content_features = get_feature_representations(model, content_path, style_path, len(style_layers))
    
    #load initial output equalling content
    init_image = load_img(content_path)
    init_image = preprocess_img(init_image)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    
    #define the optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)    
    
    best_loss, best_img = float('inf'), None   
    
    cfg = {
      'model': model,
      'alpha': alpha,
      'beta' : beta,
      'init_image': init_image,
      'style_features': style_features,
      'content_features': content_features
    }   
    
    global_start = time.time()    
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(iterations):
        
        #Compute the gradient for the loss
        grads, loss = compute_grads(cfg)
        
        #backpropagte to get optimal loss
        opt.apply_gradients([(grads, init_image)])
        
        #The values of the generated images can vary from -infinity to infinity so we clip them
        #as per the values subtracted during preprocessing
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)     
    
        #Store best loss value and subsequent image    
        if loss < best_loss:            
            best_loss = loss
            best_img = depreprocess_img(init_image.numpy())
        
        #Print current loss
        if i % print_iterations== 0:            
            print('Iteration: {}'.format(i))        
            print('Total loss: {:.4e}'.format(loss))               
                
            imgs.append(depreprocess_img(init_image.numpy()))
            
    print('Total time: {:.4f}s'.format(time.time() - global_start))    
   
    #save intermediate generated images
    for i,img in enumerate(imgs):
        Image.fromarray(img).save('output' + str(i) + '.jpg') 

    return best_img, best_loss
