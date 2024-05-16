LINK to Styling Syntax:
https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax


# 1. About our Challenge
## 1.1 About the project


### 1.1.1 The Challenge
UW-Madison GI Tract Image Segmentation:
https://www.kaggle.com/c/uw-madison-gi-tract-image-segmentation

The primary challenge here is to do segmentation of abdominaél medical MRI-Scans. We want to automatically segement anatomical structures like the stomach and intestines including the identification of the precise pixels in the image. We need to classify for the image slices the stomach, large_bowel and small_bowel and submit together with the pixels of the respective organ. 

### 1.1.2 The Problem 
In 2019, around 5 million people worldwide were diagnosed with a type of cancer that affects the digestive system. About half of these patients might receive radiation treatment. This treatment involves targeting the cancer with X-ray beams every day for a few weeks, while trying not to harm the stomach and intestines. New technology allows doctors to see exactly where the tumor and intestines are each day, which is important because their positions can change. Normally, doctors have to manually adjust the radiation equipment to make sure the beams hit the tumor and not the other organs, a process that takes a lot of time. This could take so long that what should be a quick treatment might last an hour, making it hard for patients. There's hope that using deep learning technology could speed this up by automatically adjusting the equipment.

![image](https://github.com/marioholzheu/MLFinalProject/assets/163416187/b5ba0dd8-a8eb-4e3c-9588-32f3af73bfe1)

The Dark red part shows the stomach and the pink line shows the tumor. The rainbow colors are the dose levels of radiation - green low red high. 

The UW-Madison Carbone Cancer Center, which has been using this technology since 2015, is supporting a project to improve this process. They're providing anonymous MRI scans from their patients for a competition. The challenge is to create a computer model that can identify the stomach and intestines on these scans quickly. If successful, this would help doctors deliver stronger, safer radiation doses more quickly, improving treatment for cancer patients by reducing side effects and helping them control their cancer better.

### 1.1.3 Evaluation Metrics
This competition is evaluated on the mean Dice coefficient (40%) and 3D Hausdorff distance (60%). 

- The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by: ![Screenshot 2024-04-16 at 19 38 58](https://github.com/marioholzheu/MLFinalProject/assets/163416187/fff5c072-3d1c-4098-ab92-a370c2cb3558) where X is the predicted set of pixels and Y is the ground truth. The Dice coefficient is defined to be 0 when both X and Y are empty. The leaderboard score is the mean of the Dice coefficients for each image in the test set.

- Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (In this competition, the slice depth for all scans is set to 1.) The scipy code for Hausdorff is linked. The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

### 1.1.4 The (Sustainability) Impact 
If this challenge is successfull it allows radiation oncologists to safely deliver higher doses of radiation to the tumor while avoiding the stomach and intestinies. This could speed up the treatments a lot, improve the success rate and reducde side effects. Patients would have a better outcome and doctors could focus on more important tasks and even may see more patients with the saved time. By this it would also contribute to several SDGs. 

This challenge adresses several of the SDGs like Good Health and Well-being, Industry, Innovation and Infrastructure and Partnerships for the Goals:
- Good Health and Well-being (SDG 3): This goal aims to ensure healthy lives and promote well-being for all at all ages. By developing a model that can automatically segment the stomach and intestines on MRI scans, the challenge seeks to enhance the effectiveness and safety of cancer treatments. This can lead to better patient outcomes, fewer side effects, more time for other patients and more accessible treatment options.
- Industry, Innovation, and Infrastructure (SDG 9): The challenge encourages innovation in medical technology by using deep learning to improve cancer treatment processes. The development and implementation of such advanced technologies in healthcare can enhance research and development capacity and promote sustainable industrialization. 
- Partnerships for the Goals (SDG 17): The collaboration between UW-Madison Carbone Cancer Center and the broader research community, including data scientists and technologists around the world, exemplifies a partnership that leverages resources, expertise, and technology to achieve a common goal. This cooperation is essential for sharing knowledge, technology, and resources, thereby fostering an inclusive approach to solving complex global health challenges.


## The 1.2 Data 

### 1.2.1 Existing Data 
Train Data: 
- Training Annotations: RLE-encoded amsks: single value (pixel) with count of how many times that value occurs in a row -> reduction of data
- Images: 16-bit (each pixel 65,536 shades), grayscale, PNG-Format
- Cases:
  - Multiple sets of scan slices
  - Each set is identifiable by the day
  - Cases are split by time or by case 
      - Early Days -> Train Data
      - Later Days -> Test Data
      - Case -> Entire case is in train or test
   
Test Data (same as Train plus):
- Entirely Unseen -> only accesible in Kaggle while running for submission
- About 50 Cases

### 1.2.2 Existing Files
#### train.csv 
![Screenshot 2024-04-16 at 23 25 03](https://github.com/marioholzheu/MLFinalProject/assets/163416187/6220a116-d72f-490b-aeb6-e6c8d820ee4a)

Three Columns:
- ID -> unique identifier
- Class -> 3 unique values: stomach, large_bowel, small_bowel
- Segmentation -> missing masks for some files 

#### sample_submission.csv
![Screenshot 2024-04-16 at 23 29 10](https://github.com/marioholzheu/MLFinalProject/assets/163416187/5719add9-76cd-4453-a479-29e3013163ba)

#### train folder
- 85 cases / 1 - 5 Sets / 144 images or 80 images
- Images name: (ex. 276_276_1.63_1.63.png)
  - always includes 4 numbers: slice width / height (integers in pixels) -> resolution - and width / height pixel spacing (floating points in mm) -> size of pixel 

#### test folder
- empty -> hidden test set


### 1.2.3 Data Analysis
Data Analysis well explained - good for understanding the project better but with TensorFlow: https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda -> we put some core information out of it below 

Core Informations:

#### Core Annotations:
![Screenshot 2024-04-16 at 23 54 15](https://github.com/marioholzheu/MLFinalProject/assets/163416187/fafe253b-6f92-4e7f-ba29-07f1ae84a6f0)
-> not all immages have segmentation maps for varios organs. Its possible to have 0 (21906x - 57%), 1 (2468x), 2 (10921x) or 3 (3201x) Annotations per image 

#### Image Sizes and Pixel Spacing :
Not all Images are the same size - only 4 sizes present 
- 234 x 234 (144x)
- 266 x 266 (25920x)
- 276 x 276 (1200x)
- 310 x 360 (11232x)

Two different pixel spacings:
1.5 x 1.5 (37296x)
1.63 x 1.63 (1200x)

#### Cases: 
Most Cases have 144 Slices - some 80 

# 2. Existing Solutions 
## 2.1 Overview
Most Used Models
- Unet: (2D and 2.5D Model and 3D Model with MONAI)
- DeepLabV3+
- DeeplabV3

-> Different Approaches: 
- 2D data refers to standard images, which contain height and width dimensions.
- 3D data refers to volumetric data, which might include multiple sequential images over time or space (like a CT scan), or images with depth information.
- 2.5D data often involves combining multiple 2D images taken from different perspectives or at different times to provide a more detailed or layered representation that isn't quite 3D but offers more information than a single 2D image. In the examples we saw, they normale took 3 images, so always the image before and after for attention. 

## 2.2 Sum-Up for different Code Solutions
Link to competitions: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/leaderboard
### Winner:
The 1st place solution for the 2.5D segmentation task effectively utilizes a two-stage approach: classification followed by segmentation. Here’s a summary of how their solution operates:

- Overall Pipeline:
  - Two-Stage Approach: Classification Stage: Determines whether images contain any target objects. This stage helps in filtering out images without targets, focusing computational resources on images that actually require segmentation. Segmentation Stage: Performs the actual segmentation of targets within the images identified in the classification stage.
  - Model-Weighted Fusion: Both stages employ a strategy of model-weighted fusion, which enhances the robustness of the solution and optimizes performance.
- Data Handling: Utilizes a stride size of 2 and processes three layers to create 2.5D data. This approach helps in capturing contextual spatial information across slices.
- Augmentation Techniques
  - Training Time Augmentation: Images are resized to 640x640 or 512x512. Employed techniques such as RandomCrop (to 448x448), RandomFlip, Elastic Transformation, Grid Distortion, and Optical Distortion to enhance model generalizability and robustness.
  -  Test Time Augmentation (TTA):Uses horizontal flip and weighted fusion to enhance prediction accuracy, providing a slight increase in the score.
- Model Details: Uses a U-Net architecture with EfficientNet (B4 to B7) backbones. These choices leverage the depth and efficiency of EfficientNet architectures to improve feature extraction capabilities crucial for accurate segmentation.
- Training and Inference: The classification network uses Binary Cross-Entropy Loss (BCELoss). The segmentation network employs a combination of BCELoss and Dice Loss, weighted in a 1:3 ratio to prioritize segmentation accuracy.

### Second Place:
Similar to the Winner they used a two-stage approach and 2.5D images
- Preprocessing with YOLOv5:
  - Background Removal: Uses YOLOv5 to crop the images effectively, removing irrelevant background details that might confuse the segmentation model.
  - Signal Removal from Arms: In abdominal MRI imaging, RF field inhomogeneity often results in high signal intensity around the arms. YOLOv5 helps them to exclude these areas from the image before segmentation to prevent these artifacts from affecting the normalization and segmentation processes.
- Model Details
  - Input Handling: Processes five consecutive slices at a time (s-2, s-1, s, s+1, s+2), each resized to 512x512 pixels. This multilayer slice approach helps the model capture more contextual information than a single slice could provide.
- Backbone Architectures:
  - Stage 1: Uses EfficientNet B4 and Swin Base models to classify slices as positive or negative.
  - Stage 2: Employs more robust architectures like EfficientNet L2, ConvNeXt XL, and Swin Large for enhanced feature extraction capabilities, which is crucial for accurate segmentation.
- Decoder and Loss Functions:
  - Utilizes UperNet as the decoder, which is known for its strong performance in semantic segmentation tasks.
  - Applies a combination of Cross-Entropy (CE) and Dice Loss in a 1:1 ratio to optimize both pixel-wise classification accuracy and the overlap between the predicted segmentation masks and the ground truth.
- Training Regime:
  - Trains the models for 20 epochs with Stochastic Weight Averaging (SWA) applied over the range from the 11th to the 21st epoch, enhancing the generalization of the model.


### Third Place: UNET & EfficientNet - public score 0.89 (infer) -> 3. Place Winner 
- Infer: https://www.kaggle.com/code/hesene/3rd-place-winning-solution 
- Train: not publicly available

Yaseen -> Explain the most important stuff about the solution and what it does

### Our Code Base: 2.5D U_Net - public score 0.862
- Train: https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-train-pytorch
- Infer: https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-infer-pytorch
This solution is our base and therefore explained in point "3. Our Approaches". 

### 2D UNET - public score 0.842 
- Train: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch
- Infer: https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch
This solution is working very similar as the 2.5D UNet. The main difference is that it works with 2D images instead of 2.5D images as in the previous solution. The rest of the code is very similar, therefore no addtional explanation necessary. 




## 2.2 Conclusion
Data Processing, Data Analysis and Data Visualization well explained: 
•	Just for understanding the Project better - with Tensorflow:  https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda 


# 3. Our Approaches 

We will approach the practical part as follows: 
1. Run of Unet Model (Kaggle Code) with Fold 0 and 5 Epochs
2. First Comparison: ⁠⁠Everyone in the team runs their model/backbone or 2 - 3 different models/backbones with the same hyperparameters and compare if they perform better or worse then U-Net (Only on Fold 0 and 5 Epochs)
3. Second Comparison: Everyone uses their best model and adjust hyperparameters until best parameters are found (Only on Fold 0)
Addtionally each team member will analyse one of the chosen models theoratically and show how they work.


## 3.1 About the Code Base 
Train: UWMGI: 2.5D [Train] [PyTorch] 
Infer:  -> Works only on Kaggle with hidden test data and used for final submission 

Why this code base? 
We chose a 2.5D code solution as our base because it strikes an optimal balance between achieving better results compared to a 2D solution, while requiring less computing power and runtime than a 3D approach. The code is well-structured and modular, making it highly suitable for comparing other segmentation models besides Unet and experimenting with different parameters to achieve optimal results.

How does the 2.5D code work?
The 2.5D images consist of three slices combined into one picture, as illustrated below. This process is not handled in the code; the images are precomputed and loaded directly as 2.5D images.
![Screenshot 2024-05-11 at 18 51 32](https://github.com/marioholzheu/MLFinalProject/assets/163416187/417d9227-498b-4ef4-8c77-60dfdae17ffb)

Activation Function used: 
In this example, the sigmoid activation function is used due to the multi-label nature of the segmentation task where regions of different organs, such as the Stomach, Large Bowel, and Small Bowel, may overlap within the same image. Each pixel in the image can belong to multiple classes simultaneously, and sigmoid allows for this by treating the classification of each class independently. This means the model can assign a probability for each class at each pixel without the constraints of mutual exclusivity imposed by softmax, making sigmoid ideal for scenarios where classes are not mutually exclusive and can overlap.

Folds with function StratifiedGroupFold:
In this project, the StratifiedGroupFold method is employed for splitting the data to prevent data leakage and to ensure a balanced distribution between empty and non-empty mask cases. Due to time constraints, only Fold 0 was utilized, and the training was limited to 5 epochs on the existing UNet model, which features an efficientnet-b0 backbone. This configuration serves as the foundation for our initial round of model comparison, allowing us to efficiently establish a baseline performance while managing resource and time limitations effectively.



Result of 2.5D in infer:
![Screenshot 2024-05-07 at 22 44 51](https://github.com/marioholzheu/MLFinalProject/assets/163416187/1f2a7779-3b39-4a0f-912b-e8e86243ca14)


### 3.1.1 Hyperparameters of Unet Model -> Mario: 

The following hyperparameters from the file were used as listed: 

![Screenshot 2024-05-11 at 18 27 53](https://github.com/marioholzheu/MLFinalProject/assets/163416187/6456b629-a323-4140-ae98-bd1417a64589)
 

### 3.1.2 Training Results of Unet Model -> Mario: 



## 3.2 New Model Approaches and Comparison 

Comparison: 
--> FILL TABLE IN PPT IN TEAMS --> Screenshot here 

### 3.2.1 UNet++ - Mario
Also tried PAN - due to 3 times higher computation I stopped this experiment as for 1/5 of the data and 5 Epochs Training would be over 7 - 8 hours. 

Why Unet++
According to this paper https://arxiv.org/pdf/1807.10165, which compares the U-Net with the UNet++, the Unet++ gets quite better scores for all different Datasets used. Therefore it seems really promising. 
![Screenshot 2024-05-11 at 22 00 45](https://github.com/marioholzheu/MLFinalProject/assets/163416187/053f1362-c827-45df-950f-a5adff82eea9)



#### Comparison of Unet and Unet++

Quantitative Results: 
- Purple: Unet
- Pink: Unet++ -> Higher Dice and Jacccard  
-> Result: Dice is 0.02 better after 5 Epochs
![Screenshot 2024-05-11 at 23 35 05](https://github.com/marioholzheu/MLFinalProject/assets/163416187/8fc10ccc-ee43-41aa-b784-e9b1ccc8d61c)

Qualitative Results: 
<img width="907" alt="Screenshot 2024-05-14 at 17 45 33" src="https://github.com/marioholzheu/MLFinalProject/assets/163416187/629930f7-330a-47d0-a430-a63bcaed1b0b">

Unet++ 
Scheduler eLR vs cosine 

![Screenshot 2024-05-12 at 12 24 41](https://github.com/marioholzheu/MLFinalProject/assets/163416187/54cb8e81-1b12-432d-9328-f7d15d4832de)





#### Unet++
##### Description of Unet++ 

##### Possible Enhancements
Backbone / Encoder: 
Efficientnet with more parameters 
resnet200
densenet




##### Paper Summary:
Comparison U-Net and UNet++ with and without deep supervision https://arxiv.org/pdf/1807.10165
Skip Connections Redesign: https://arxiv.org/pdf/1912.05074v2
Language meets Vision Transformer in Medical Image Segmentation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7329239/
UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATIONhttps://arxiv.org/pdf/2004.08790v1
#### 

Model Architecture of Unet++: 




### 3.2.2 MODEL - Yaseen 
Chosen Model: 

Model Architecture of (CHOSE ONE)

### 3.2.3 MODEL - Shiva
Chosen Model: 

Model Architecture of (CHOSE ONE)

### 3.2.4 MODEL - Nikhila
Chosen Model: 

Model Architecture of (CHOSE ONE)



## What other possible solutions could be tried? 


# 4. Conclusion: 





# DELETE BEFORE SUBMISSION Extras: 

### UNET - 3D Solution with MONAI (based on 2.5D U_net solution)
- Infer: https://www.kaggle.com/code/yiheng/3d-solution-with-monai-infer

Yaseen -> Explain the most important stuff about the solution and what it does

Comment Mario: I think drawbacks here can be deleted then? 
Drawbacks: 
Ambiguity for hyperparameters, not specified as model uses different pretrained models. To make changes in that model we will have to delve deeper into 10 or so pretrained models with pre initialized weights and hyperparameters. 
Code difficult to implement if wamtedto make any changes. A very real possibility that the hidden models used for different layers of the solution are pretrained with hidden hyperparameters for the winning solution on 3rd place. 
Currently looking into 2.5D, Unet, Unet++ and 3dUnet. 


### PSP,UET,DEEPLAB,SwinUnet
Comment Mario: add public score? 
https://www.kaggle.com/code/masatomurakawamm/uwmgi-pspnet-u-net-deeplabv3-swin-unet

Model runs correctly on Kaggle. However still needs to be configured to run locally on colab.
Model trained and tested: Deeplabv3
Loss: Focal Loss
Optimizer: Adam
Lr : 1e-4

Other than that differnet models are built with their own settings such as lr for Swin-Unet is 1e-3 and loss is given in the following manner :

     optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy', losses.dice_coef]

              
A Still to be reviewed in what aspects are the other models being used. It is likely the models are initialised and deleted at the end. The only model run is deeplabv3 possibly due to alck of resourcs to run all models. Code provides a nice abseline to try and run the other odels with the same data preparation since the models' architecture has been defined already. 

Suggested changes: Set-up training loop to run for different models, one-by-one considering the GPU resources. 
Point to be noted: Model runs in original development environment. Not experimented on the the most up-to-date environment.

















