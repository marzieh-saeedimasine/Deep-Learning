Deep Learning modeling Repo:  

This repository provides a guide for deep learning modeling of different computer vision projects  

**Overview of deep learning modeling for Mnist digits classification project**    
-- 70000 digit images from Mnist dataset have been used for multi class classifications  
-- Dense neural network has been tested for image classification     
-- Convolutional neural network (CNN) used for modeling
-- Model Quntization: Quantize model weight using tf.Lite or quantize each model layer using tensorflow_model_optimization, compare the model size  
     

**Overview of Transfer learning for flower image classification project**    
-- 3500 flower photos have been used for multi class classification of flowers between 5 groups:  
 ("roses", "daisy", "dandelion", "sunflowers", "tulips") each containes about 700 images  
-- convolutional neural network (CNN) and data agumentaion has been tasted for classification of flowers  
-- Transfer Learning with pre-trained model of fishes download from tensorflow.hub used for flower image classification  
-- Transfer learning with Inception model used for flower image classification  

**Overview of Transfer learning for image classification project**    
-- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class has been used for image classification project  
-- Convolutional neural network (CNN) and data agumentaion has been tested for image classification    
-- Transfer learning with ResNet50 model has been used for image classification    
-- Transfer learning with MobileNetV2 model has been used for image classification    
-- compare accuracy and performance of ResNet50 and MobileNetV2 in image classification  

**Overview of Transfer learning for tomato leaf disease calssification project**   
Deep learning modeling of tomato leaf disease detection  
Data were taken from : https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf  
-- Convolutional neural network (CNN) and data agumentaion has been tested for tomato disease detection  
--  Transfer learning with MobileNetV2 model has been used for tomato leaf disease calssification   
-- compare accuracy and performance of Transfer learning and pure CNN model in classification  

**Overview of Vision Transformer (ViT) modeling of image classification project**   
-- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class has been used for image classification project    
--Vision Transformer (ViT) has been applied for CIFAR image classification. Model consists of the following key components:  
-- Patch Embedding: Converts an image into a sequence of flattened patches.  
-- Position Embedding: Adds positional information to the patches.  
-- Transformer Encoder: Applies multiple layers of self-attention and feed-forward neural networks.  
-- Classification Head: Maps the final representation to the class labels.  
-- compare the accuracy and performance of ViT with pure CNN and other transfer learning in TransferLearning-CIFAR-Project.ipynb 


**Overview of object detection and localization project with different models**     
-- Object Detection model on images  
-- Tensor Flow object detection models that have been trained on the COCO 2017 dataset, from TensorFlow Hub   
-- object detection with YOLO  
--compare two models Mask R-CNN Inception and YOLO in object detection  


**Overview of deep learning models**  
-- ResNets50 consists of 50 layers of identity_block and Convolutional_block with skip connections
-- UNet model (U-shape) consists of an encoder (downsampler) and decoder (upsampler) with a bottleneck in between. It uses the skip connections that concatenate encoder block outputs to each stage of the decoder. 
