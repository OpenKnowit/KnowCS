# Semantic augmentation of images for Computer Vision Tasks

COMP 4971F – Independent Work (Spring 2020) 

Minsoo, Khang (mkhang@connect.ust.hk) [Supervised by Dr. Desmond Tsoi] 

Abstract – Computer Vision tasks (such as object detection) often require a large amount of image dataset for a deep learning solution with satisfactory performance. In the event such large number of images are not available, many researches often resort to image augmentation such as: random flipping, color jittering, etc. However, these conventional augmentation techniques are often restrictive in terms of variations which augmentation could bring to the image dataset. As such, this research proposes a different approach to image augmentation to create a wider variety in images generated whilst still preserving the semantics of the original image. 

# Table of Contents

Introduction ..... 

Goal of research study........... 

Capturing semantics in images ..... 

Methodology – Image reconstruction with latent vectors .... 5 

Methodology variations .... 

Transferring background information to other images ........... 

Discussion and limitations ..... . 11 

Conclusion and future studies....... . 12 

Appendix ............ .... 12 

Code for reference ........ ... 13 

Code for data generation described in Figure 6 ..... .13 

Code for testing transfer of background described in Figure 13 and 14... ..14 

# Introduction

Computer Vision tasks such as object detection often relies on deep learning as the main solution architecture. Taking object detection as an example, popular deep learning solutions such as YOLO (You-Only-Look-Once) or RetinaNet require a large amount of labelled image dataset for satisfactory performance. The exact minimum number of images required vary for different applications but in general ranged from hundreds to even thousands of images per object class needed for detection. 

Curation of such a large number of images is certainly a labor-intensive task and if the images were to involve sensitive information (such as facial images), curation of large dataset could become difficult. To narrow down the scope of this study, this report will focus only of image dataset of facial images, but the technique utilized could be transferred to other target objects for different object detection / computer vision tasks. 

There had been similar studies carried out previously in the field of image completion for facial images using the generative approach [1]. However, generative approach as discussed in [1], requires a very computationally demanding model – requiring an encoder-decoder model, two discriminator models and one pretrained parsing network. This research aims to explore into an alternative model with UNet without any discriminator models, making it less computationally demanding. 

# Goal of research study

This research aims to propose a new image augmentation pipeline developed on top of UNet deep learning architecture. Image augmentation in general aims to provide images that are same in semantic meaning (such as image of a dog/cat) while looking different from the original image and this is a common workaround to generate larger training image dataset. With the focus on facial images, this study proposes a pipeline to generate facial images which look visually convincing to be a facial image while looking variant from the original image for the purpose of augmentation. 

# Capturing semantics in images

Conventional image augmentation techniques (such as random flipping) aims to provide variation to the original image whilst still preserving the semantics of the original image. In Figure 1 below, the two images both represent the meaning of “Dog” while one had been horizontally flipped from the other. Such augmentation technique preserves the semantics of the image (that it is a dog) as the augmentation was largely restrained within the given image. 

![](images/fd9e19aa67825695ccea81319ab1f4050d89a6c02349753fd19e386d174698c4.jpg)


![](images/c34c6243ad0390743a51d1b01307d52e6f939d93dcb359ca1af43edc0346874c.jpg)



Figure 1, Example of dog image that had been horizontally flipped for augmentation – Online Source [2]


However, there are many aspects of the image which are not necessary for humans to identify it as an image of a dog. For example, the pointy shape of the dog’s ear is not a feature unique to dogs and humans are able to identify it as a dog even without it. Removing such unnecessary details in the augmentation process could allow greater variety in the images produced. In another words, fine detailed information such as pointy ears in the dog image is not necessary to be fixed during augmentation and if such features could be varied, the augmented image created could have a wider variety (such as having round / pointy / flat ears). 

To achieve such unconstrained augmentation (Semantic augmentation), the augmentation process was experimented to operate on a higher-level representation of the image instead of the image directly. This means, the images (such as dog image in Figure 1) is converted into a higher-level representation (such as latent vectors) for augmentation purposes instead of applying augmentation directly on the images. Such representation of images as latent vectors is a well-known technique used in the application of Content-Based-Image-Retrieval (CBIR) [3]. 

To convert the images into corresponding latent vectors, a conventional UNet model was modified to incorporate latent vectors of a fixed dimension in the bottleneck layers. As shown in Figure 2, the input image is compressed down to a N-dimensional latent vector before it is decompressed back to recreate an image as similar to the input image as possible. (As this study proposes an augmentation pipeline involving 2 UNet models, the UNet model proposed for generation of latent vectors shown in Figure 2 would be named as primary Unet) 

![](images/b5b59195b15518d36fd7642014cc3b4e735ba07d5006570be42c5d2d51f5cd1f.jpg)



Figure 2, Architecture of the primary UNet model with introduction of N-dimensional latent vector in bottleneck layer – diagram modified from [4]


While the functionality of the primary UNet model architecture shown in Figure 2 could be deemed as trivial (as all it does is identity image creation with no meaningful operation), it allows for an input image to be converted to its corresponding latent vector as a side product. In another words, after the model described in Figure 2 had been trained, the encoding half of the model could be extracted such that now the model is able to produce a N-dimensional latent vector for any input RGB image. Figure 3 below illustrates the idea of using the encoding-half of UNet as a generator of latent vectors. 

![](images/11840474e1e788cdd1e96ab49f23598c6840f8974cb3acc0493fe4e3b705f4fa.jpg)



Figure 3, Illustration of the encoding half in the Identity mapping UNet model described in Figure 2 previously


The UNet model was configured with hyperparameter setting of – Batch size 32, Learning rate 7.5E-4 and latent vector dimension of 100. The model was trained on the publicly available CelebA dataset [5] and had managed to converge with a low validation loss of 4.72E-4 within 5 epochs of learning. Upon convergence, the identity mapping primary UNet model was evaluated qualitatively to ensure that the model was indeed able to produce an output image as similar to the input image as possible. Figure 4 displays some of the input and output image of the identity mapping UNet model. 

![](images/d3831f1901362b192d46c5e18aadc01746ecfa88dd7a6d65bf9eef3d83cdca35.jpg)


![](images/5c56176ad44756502681f79c0ee681bf980e0ccc8740f90a7631326bae8c4fe6.jpg)



Figure 4, Input celebrity image (left) and output image from identity mapping UNet (right)


Upon closer inspection across different celebrity images found in CelebA dataset, it was noted that whilst the facial details and textures could mostly be reconstructed, certain color tones encountered distortions as shown in Figure 5. Despite training the UNet model for a longer period of time, color distortion was present to a small extent across different images and as the main focus of this UNet model was to generate a latent vector that encapsulates as much of the input image information as possible instead of the actual identity image mapping, this model was used for latent vector generation. 

![](images/5a85c0b8bb4d4636975b993fa50a4a2440c28aa9dc26284f2ff01a0c88f0694c.jpg)


![](images/1c22908c5457322f2bb289bdcd4f651c953436f88ad33f7b085966f718cffadf.jpg)



Figure 5, Input celebrity image (left) and output image from identity mapping UNet (right)


# Methodology – Image reconstruction with latent vectors

To keep in line with the interest of generating augmented images that are variant from original facial image while still looking visually convincing to be a facial image, the input image was split into three different forms of input before channeling through a new UNet model (secondary UNet model) for image reconstruction. A given facial image is split into three types of input: heavily blurred image, latent vector of facial region and latent vector of the remaining background (Non-facial region). This splitting of image can be visualized in Figure 6 below. 

![](images/de06fd1b3b126bfabb7c281376a776108248f14a4f747cc2b3d9363de2759894.jpg)



Figure 6, Flow diagram of different inputs to the secondary UNet model (2 latent vectors and blurred image) *Encoding-half UNet refers to the encoding half of primary UNet described in Figure 3


After the original celebrity image has been processed into three different types of input data for training, the facial image augmentation UNet model (secondary UNet) is trained to take in the blurred image along with the two latent vectors in the bottleneck layers, before outputting an image as similar to the original celebrity image as possible. Figure 7 below is an illustration of how a blurred image of a celebrity image, along with the two previously obtained latent vectors (one for facial region and another for the remaining background) could be input into a UNet model and trained to generate a photo-realistic celebrity image. 

![](images/4880278b2d3dd0b9239a35d68942bdcf78d1b7f9fb9fd5958278993a301d1cf2.jpg)



Figure 7, Data flow diagram of blurred image and two latent vectors in training a secondary UNet model for facial image augmentation


The secondary UNet model was trained with 180,000 celebrity images while 10,000 was set aside for validation and another 10,000 images for testing purposes. The UNet model was trained in a google colab environment with a Tesla P100 GPU and had managed to converge with validation loss of 0.0089 within 5 epochs at a total training time of around 1 hour 40 minutes. 

# Methodology variations

Prior to implementing the methodology of using the blurred target image and two latent vectors (described in Figure 6 and 7), other variations in methodology were explored. While the methodologies vary, the core intention remained the same – creating augmented images that are variant from the original facial image while remaining visually convincing to be realistic. During the phase of preliminary testing, as there were multiple studies carried out for image completion / synthesis for the facial region, such as in [1], this research started off with the aim of image completion / synthesis for background region of a facial image. 

The rationale for this direction was, aside from the facial region (eyes, nose and mouth), image background also plays a crucial role in data augmentation. For example, varying the individual’s hair style and the background setting could certainly generate more images of greater variety in augmentation. If such goal could be achieved for background images, it could then be merged with techniques targeted at facial regions (such as [1]) and provide a complete facial image augmentation pipeline. 

With the initial goal of augmenting the non-facial region of the facial images, data flow in the secondary UNet model training had been initially designed to be different from Figure 7 as shown in figure 8 below. 

![](images/b3b5fec43f870c541e510135b20850801ac5598770e35887c7aeaf6e8fa6e89b.jpg)



Figure 8, Data flow of secondary UNet model with two input streams



1 – Celebrity image with only facial region, 2 – latent vector representing only the background information of image


What made the dataflow design and training described in Figure 8 differ from that of Figure 7 is that, the UNet model is trained to generate non-facial region information (hairstyle and background) solely based on the facial region and a latent vector representation of the background. However, based on preliminary testing, upon convergence, it was noted that there was a limit to which the model could provide meaningful inference on the non-facial region pixels. Figure 9 is the model’s prediction of the non-facial region pixels of the same celebrity image shown in Figure 8. 

![](images/8d54050d511363accf93bc22ca73eef5e71920874d7c417c5d85ebedf7bcd76f.jpg)



Figure 9, Target celebrity image (Left), non-facial region generated image (right)


From the model output captured in the right image of Figure 9 there were certain key aspects to note. Firstly, for regions such as the hair, the model was able to provide meaningful prediction whereby the hair generated does not look random and yet different from the original celebrity image which is the aim of this augmentation process. However, for the background of the image, while it looks different from the original celebrity image, it does not have any meaningful appearance (just pure grey in color). 

Upon observing across different images, it was noted that the background pixels (behind the human individual) of images generated by the model were all grey in color. The possible reason behind this observation could be due to variability difference between the human hair and the image background. This means, the UNet model was able to provide meaningful prediction for the hair region as there was a set of fixed structure that could be inferred across different celebrity images (such as brown in color and always observed to be encapsulating the facial region of a human). 

However, for the background region of the image, there was no fixed structure and the variability was too high for the model to provide any meaningful prediction during runtime. For example, the celebrity image captured could be in a park, household, coffee shop or office background setting and as the variation is endless the model is not able to provide meaningful prediction solely from the latent vectors shown in Figure 8. As such, the challenge was to balance between providing minimal information to the UNet model so as to obtain an image variant from the original image, while providing sufficient information so that the model is able to provide a photo-realistic and meaningful facial image. 

To strike this balance, this research explored into the possibility of applying gaussian blurring across the entire facial image instead of excluding any specific region as the latter would be too extreme and the model could face difficulty in making meaningful inference (shown in Figure 9). As such, as described in Figure 6, during the dataset preparation, the celebrity images were blurred twice using a 45 by 45 kernel with standard deviation of 45. Figure 10 shows that gaussian blurring twice was qualitatively better than just once, in terms of minimizing the details remaining in the image prior to channeling into the secondary UNet model. 

![](images/fcb824c85d4f53d397954f68de6d21d2363a97f7cc68f640264f5d24fc043bc6.jpg)


![](images/f30cf4e335f436222312f8e8b5d27013337aeb12a783f0447358c360b4879264.jpg)


![](images/ec59edf873e34d10f980907978ee155a543300f1aa3318f85b57991ad8d544f3.jpg)



Figure 10, Original celebrity image (Left), Gaussian blurring once (middle), Gaussian blurring twice (right)


As shown in the middle image of Figure 10, gaussian blurring once still retained certain details in the eye’s region of the facial image while blurring twice (right image of Figure 10) had managed to remove such details. This limits the amount of detail information provided to the model, so that it would be able to produce images with greater variation. Upon convergence, the model was able to provide a more photo-realistic, yet different image as shown in Figure 11. 

![](images/375bb5801f25fdce25654b2a570d161668493bb364b90548aead99e6bc0c8737.jpg)



Figure 11, Original celebrity image (Left), Gaussian blurred input image (Middle), Model output generated image (Right)


Compared to the model’s generated image shown in Figure 9, the image generated in Figure 11 had certainly managed to capture certain information regarding the background of the image and still generate a facial image that, though slightly blurred, still look photo-realistic and visually convincing. 

# Transferring background information to other images

While the data flow illustrated in Figure 6, 7 and image generated in Figure 11 shows that gaussian blurring and latent vectors allow photo-realistic yet different looking images to be generated, it still limits the image augmentation by fixing the human individual in the image to that image’s background. In another words, the question was: “Could a person in an image be placed in another image’s background and yet look realistic?”. This would create even greater variant in images generated and delved into the area of transferring the background information between images. To experiment if the data flow and model training proposed in this research study could accomplish transferring of background information, two celebrity images were chosen as shown in Figure 12. 

![](images/a739c485acde6f6d6709796dee3dfa0d4c514a07c5485e34758216b121b3eef0.jpg)


![](images/fe90c03b655e3d59fde603ce8ba9b801f9c46cde568a3b9d36858e003d2c093a.jpg)



Figure 12, Celebrity A image (Left) and Celebrity B image (Right)


The goal was to augment celebrity B image such that while the facial region looks realistic yet different, the background would be transferred to be the same as that of celebrity A image. To achieve this, the data preparation flow discussed in Figure 6 and 7 were carried out as shown in Figure 13. 

![](images/1a4af3a5d3771baa6f89f89242194790a5b38cb95eb172cf9c9852c41d739f26.jpg)



Figure 13, Data flow of combining facial image of celebrity B with the background region of celebrity A


The previously obtained secondary UNet model was NOT retrained and instead the same trained model was used with the new data input described in Figure 13. Despite not having a ground-truth image to learn from, the model was able to generate image that was photo-realistic variation of celebrity B whilst transferring the background information from celebrity A image as shown in Figure 14 below. 

![](images/146568ec32cb54ec0a6e1b0631c79b3a147d073f7d475e736b291e4bd007d499.jpg)



Figure 14, Comparison of background information transfer between images Celebrity B image with background transfer from celebrity A image (Left) Celebrity B image with original background information (Right)


# Discussion and limitations

From the results shown in Figure 11 and 14, it is clear that the method proposed of applying gaussian blurring twice is effective in balancing the need to provide minimal information to allow variance in image generated whilst providing sufficient information for meaningful inference. Additionally, as shown in Figure 14, when the scale (aspect ratio and coverage) of facial region in the two images are similar, the background of one image could be transferred to another through the proposed gaussian blurring and latent vector pipeline. 

Upon further investigation, it was noted that for background information transfer between images, the generated image is not photo-realistic when the scale (aspect ratio and coverage) of facial region of an image differs or when the orientation of faces is not aligned between images. Examples of each of the two observations are illustrated in Figure 15 and 16 respectively. 

![](images/4cd2e140a1d798b55ebd894191cf93e4ee155e472db9479a35f42fb9b6182a01.jpg)



Figure 15, Two original celebrity images (1st, 2nd), manually combined (3rd), Gaussian blurred twice for input (4th), Model generated image (5th)


From Figure 15, two celebrity images were chosen such that the percentage of (128x128) image covered by the facial region (including the hair) differed clearly. When these two images were combined by manual cropping with pasting followed by gaussian blurring and model prediction, the generated image can be obtained as observed in the 5th image of Figure 15. The positive outcomes are that firstly, the background information could be transferred from the 1st celebrity image whilst preserving the collar color of the 2nd celebrity image. Secondly, the facial region generated (eyes, nose and mouth) is qualitatively realistic whilst looking different from the original 2nd celebrity facial region. However, the drawback is that, the image regions near the hair does not look realistic and this is potentially due to significant difference in scale of hair pixels between the two celebrity images. 

![](images/36ce53c81f3fcad4b5d381d6d322157e1f613f4ac7db626e081f3bd16a70e27c.jpg)



Figure 16, Two original celebrity images (1st, 2nd), manually combined (3rd), Gaussian blurred twice for input (4th), Model generated image (5th)


As shown in Figure 16, even when the orientation of the 2nd celebrity face is different (not an upfront portrait facial image), the new facial region (eyes, nose and mouth) generated are able to handle such difference and still provide visually convincing image as shown in 5th image of Figure 16. However, the image generated as a whole is not photo-realistic especially near the hair regions of the image. This is potentially because, as the facial orientation is not the same between the two celebrity images, whilst the eyes, nose and mouth regions could be handled, the hair region generated is not well-aligned resulting in an artificial looking image. 

# Conclusion and future studies

In conclusion, this research had managed to propose an augmentation pipeline that is able to generate facial images which are visually convincing whilst still variant from original image. While there are still room for improvement such as sharpening of generated image and handling different facial image specifications discussed in Discussion and limitations, based on the output image generated in Figure 11 and 14, the augmentation pipeline clearly shows great potential for further research. 

Possible future studies include investigating a quantitative method to measure the realistic aspect of the generated image. In this research, all images generated were largely evaluated on its photo-realism in a qualitative manner. While the current trend of research is to use a discriminator model (such as in GAN) to evaluate the “realism” aspect of the generated image, such methodology is not only computationally heavy but also does not guarantee that the generated image would always be photo-realistic. As such, researching into a quantitative metric for evaluating generated image could be a potential direction for future research. 

# Appendix



[1] Y. Li, S. Liu, J. Yang, and M.-H. Yang, “Generative Face Completion,” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 





[2] “Improving Deep Learning Performance with AutoAugment,” Google AI Blog, 04-Jun-2018. [Online]. Available: https://ai.googleblog.com/2018/06/improving-deep-learningperformance.html. [Accessed: 01-Jun-2020]. 





[3] Schettini R., Ciocca G., Gagliardi I. (2009) Feature Extraction for Content-Based Image Retrieval. In: LIU L., ÖZSU M.T. (eds) Encyclopedia of Database Systems. Springer, Boston, MA 





[4] Imgur, “3D UNet Architecture,” Imgur, 08-Dec-2017. [Online]. Available: https://imgur.com/gallery/7FHkQ. [Accessed: 01-Jun-2020]. 





[5] Liu, Ziwei et al. "Deep Learning Face Attributes in the Wild." Proceedings of International Conference on Computer Vision (ICCV). 





[6] UNet Model development skeleton referenced from: https://github.com/zhixuhao/unet/blob/master/model.py 



# Code for reference

Code for UNet model architecture had been reference and developed on top from: https://github.com/zhixuhao/unet/blob/master/model.py 


Code for data generation described in Figure 6


```python
class Input_gaussian_blurring(Sequence):
    #configured to:
    #Batch size of 32 and latent vector of size 100
    ######
    #This generator outputs three types of data as discussed in Figure 6 of the report
    # 1. Latent vector (size 100) of cropped face image
    # 2. Latent vector (size 100) of image without facial region
    # 3. Complete celebrity image that had been blurred with Gaussian blurring (kernel size 45 with sigma 45) twice
    ######
    def __init__(self, celebrity_image_list, array_of_vector_of_image_without_face, array_of_vector_of_cropped_face):
    self.celebrity_image_list = celebrity_image_list    #List of celebrity images for input number 3
    self.array_of_vector_of_image_without_face = array_of_vector_of_image_without_face    #Array of latent vectors for input number 2
    self.array_of_vector_of_cropped_face = array_of_vector_of_cropped_face    #Array of latent vectors for input number 1

    def __getitem__(self, idx):
    img_id = idx * 32
    length = min(32, (len(self.celebrity_image_list) - img_id))
    ## 
    batch_input = np.empty((length, 128, 128, 3), dtype=np.float32)    #Initializing array for batch input of gaussian blurred images
    batch_input_latent_1 = np.empty((length, 100), dtype=np.float32)    #Initializing array of latent vectors of size 100
    batch_input_latent_2 = np.empty((length, 100), dtype=np.float32)    #Initializing array of latent vectors of size 100
    batch_output = np.empty((length, 128, 128, 3), dtype=np.float32)

    for i_batch in range(length):
    image_label = self.celebrity_image_list[img_id + i_batch]

    input_celebrity_image = cv.imread(image_label)    #Reading a celebrity image to start gaussian blurring
    ###
    offsetting the image to preserve aspect ratio before resizing image to 128 x 128 x 3
    ###
    offset = (input_celebrity_image.shape[0] - input_celebrity_image.shape[1])/2
    initial_offset = int(1.4 * offset)
    end_offset = int(0.6 * offset)
    ###
    Adjusted_input_image = input_celebrity_image[initial_offset:-end_offset,:,:]
    input_image = cv.resize(Adjusted_input_image, (128, 128))
    output_image = input_image.copy()
    ###
    #First round of gaussian blurring with kernel size of 45 with sigma of 45
    input_image = cv.GaussianBlur(input_image, (45, 45), 45)

    #Second round of gaussian blurring with kernel size of 45 with sigma of 45
    input_image = cv.GaussianBlur(input_image, (45, 45), 45)
    ###
    batch_input[i_batch, :, :, :] = input_image / 255.
    batch_input_latent_1[i_batch,...] = self.array_of_vector_of_image_without_face[img_id + i_batch].copy()
    batch_input_latent_2[i_batch,...] = self.array_of_vector_of_cropped_face[img_id + i_batch].copy()
    batch_output[i_batch, :, :, :] = output_image / 255.

    return [batch_input, batch_input_latent_1, batch_input_latent_2], batch_output

def __len__(self):
    return int(len(self.celebrity_image_list) / float(32)) 
```


Code for testing transfer of background described in Figure 13 and 14


```julia
new_test_img1 = cv.imread(celebTestPath_blurred[34])
new_test_img2 = cv.imread(celebTestPath_blurred[35])
offset = (new_test_img1.shape[0] - new_test_img1.shape[1])/2
initial_offset = int(1.4 * offset)
end_offset = int(0.6 * offset)

###
new_test_img1 = new_test_img1[initial_offset:-end_offset, :,:]
new_test_img1 = cv.resize(new_test_img1, (128,128))
new_test_img2 = new_test_img2[initial_offset:-end_offset, :,:]
new_test_img2 = cv.resize(new_test_img2, (128,128))

###
#Display the original images from dataset
plt.figure()
f, ax = plt.subplots(1,2)
ax[0].imshow(new_test_img1[...,-:-1])
ax[1].imshow(new_test_img2[...,-:-1])

###
#Manual cropping and paste of background prior to gaussian blurring
new_test_img3 = new_test_img2.copy()
new_test_img3[:,:20,:] = new_test_img1[:,:20,:]
new_test_img3[:,-15,::] = new_test_img1[:,-15,:]
plt.figure()
f, ax = plt.subplots(1,2)
ax[0].imshow(new_test_img3[...,-:-1])

###
#Gaussian blurring of the manually cropped and pasted image
new_test_img3 = cv.GaussianBlur(new_test_img3, (45,45),45)
new_test_img3 = cv.GaussianBlur(new_test_img3, (45,45),45)
new_test_img3 = np.expand_dims(new_test_img3/255., axis=0)
ax[1].imshow(new_test_img3[0][...,-:-1])

###
#Model's reconstruction and generation of the image with the transfer of background information
test_prediction = unet_cropped_face_train.predict([new_test_img3, input_latent1[2:3,...], input_latent2[3:4,...]])
plt.figure()
plt.imshow(test_prediction[0][...,-:-1]) 
```

<matplotlib.image.AxesImage at 0x7f022ofdc668> <Figure size 432x288 with 0 Axes> 

![](images/430e7f56883a144ad45b5921b96569ffda95b60ed1e84f10a3fb323a79312c8a.jpg)


<Figure size 432x288 with 0 Axes> 

![](images/c1fd614691919a62b0fd8bc8e0d634b3d3a4192b5fc3134bb94ca11c36bcc607.jpg)


![](images/7d6f9cd4e74e72e1f970e02d44fc188a881d4a76b5270bccfcbafbd45eb4201c.jpg)
