Team Members: Mario Holzheu, Nikhila Srigiri, Manchinganti Sivarchan, Yaseen Mujahid

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
- 
Pre-processing with EfficientDet-D0:
- Used 5-Fold Cross Validation
- Size: 256
- Epoch: 5
- Default parameters for other settings.
- Generated training samples via image preprocessing and relabeled bad bounding boxes.

Classification
- 35 Epochs, LR 3e-4 or 5e-4, BCE loss 
- Model: Unet (Smp-Unet and Timm-Unet with different backbones and classification branches):
  - Smp-Unet efficientnet-b7, size=320, slice=5
  - Timm-Unet efficientnet-v2-l, size=320, slice=3
  - Timm-Unet efficientnet-v2-l, size=320, slice=5
  - Timm-Unet efficientnet-v2-m, size=352, slice=5
  - Timm-Unet efficientnet-b7ns, size=320, slice=5
Segmentation:
- Epochs 35, ComboLoss(BCE and Dice), all 5 slices, only positve samples from classification used, 
- Model: Timm-Unet with EfficientNet backbones.
  - Timm-Unet efficientnet-v2-l, size=384
  - Timm-Unet efficientnet-v2-l-1, size=416
  - Timm-Unet efficientnet-v2-l-2, size=416
  - Timm-Unet efficientnet-v2-m, size=416



### 2.5D U_Net - public score 0.862
- Train: https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-train-pytorch
- Infer: https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-infer-pytorch
This solution is our base and therefore explained in point "3. Our Approaches". 

### 2D UNET - public score 0.842 
- Train: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch
- Infer: https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch
This solution is working very similar as the 2.5D UNet. The main difference is that it works with 2D images instead of 2.5D images as in the previous solution. The rest of the code is very similar, therefore no addtional explanation necessary. 

## 2.3 Conclusion:
- Unet Models might still be one of the best appraoches possible. 
- 2.5D Solution might be the most promising one for this challenge for good results
- Backbones with more Parameters like Efficient-Net B7 achieve better results
- Training could happen on lower LR and higher Epochs, but we probably have not the computation power for this
- We should select the 2.5D U_net File as it has great modular code available, uses 2.5D Dimensions and also can be changed to many different models and trained on them according to the models and encoder listed here: https://smp.readthedocs.io/en/latest/models.html. SMP Solutions were also part of one of the winner solutions. Furthermore we have the possibility to link it to WANDB for visualizations, logging and saving of models.  


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


### 3.1.1 Hyperparameters of Unet Model: 

The following hyperparameters from the file were used as listed: 

![Screenshot 2024-05-11 at 18 27 53](https://github.com/marioholzheu/MLFinalProject/assets/163416187/6456b629-a323-4140-ae98-bd1417a64589)
 

### 3.1.2 Training Results of Unet Model: 
![Screenshot 2024-05-19 at 11 24 08](https://github.com/marioholzheu/MLFinalProject/assets/163416187/ea10ebc1-1eb4-4a9f-859f-3dd9ce88b29c)



## 3.2 New Model Approaches and Comparison 

Comparison: 

![Screenshot 2024-05-19 at 22 30 29](https://github.com/marioholzheu/MLFinalProject/assets/163416187/d7b814e4-423d-4431-87e0-9fbd6d5feca1)


### 3.2.1 UNet++ - Mario

Why Unet++
According to this paper https://arxiv.org/pdf/1807.10165, which compares the U-Net with the UNet++, the Unet++ gets quite better scores for all different Datasets used. This can be achieved due to nested and dense skip connections. Therefore it seems really promising. 
![Screenshot 2024-05-11 at 22 00 45](https://github.com/marioholzheu/MLFinalProject/assets/163416187/053f1362-c827-45df-950f-a5adff82eea9)

I also tried to experiment with PAN Model additionally, as training time was way higher and results seemed not better, I stopped this approach right away due to time reason and to focus more on Unet++. 

#### Comparison of Unet and Unet++
The same hyperparameter were used to see if the model itself can already outperfom unet which was the case. 
Quantitative Results: 
- Purple: Unet
- Pink: Unet++ -> Higher Dice and Jacccard  
-> Result: Dice is 0.02 better after 5 Epochs
  
![Screenshot 2024-05-11 at 23 35 05](https://github.com/marioholzheu/MLFinalProject/assets/163416187/8fc10ccc-ee43-41aa-b784-e9b1ccc8d61c)

#### Unet++ Approaches:
Tried Hyperparameters: 
- Backbone EffiecientNet-b0, b2, b4
- Batch Size: 32 and 64 depending on Backbone
- Epochs 5 - 10
- LR: 2e-3, 2e-4 and 9e-3 (This one I got with a function to find_the_best_lr see below and in code)
- Scheduler: CosineAnnealingLR, CosineAnnealingWarmRestarts and ReduceLRonPlateau (The last two did not make sense in my case as their advantage didnt come into play if just trained with 5 - 10 epochs as it was still learning
- Loss Function: BCE and Dice Loss with 1:3 Ratio and ...


![image](https://github.com/marioholzheu/MLFinalProject/assets/163416187/77fc7031-f992-4851-bc82-b3dfe7af0750)

In the run where I tried this learning rate, the model still learned in Epoch 10. Till then this approach did not outperform others. So its hard to say without training on like 15 - 20 Epochs if this would have achieved my best results. Till Epoch 10 it didnt. 

#####Best Combination:
- Backbone b4, BS 32, Epochs 10, LR: 2e-3, CosineAnnealingLR, BCE and Dice Loss 1:3

Training Processes Comparison 
The graph shows some of the differenet models I trained with
Color:
- Pink     Unet++, efficientnet-b4, lr 0.002, Dice & BCE Loss 1:3
- Red      Unet++, efficientnet-b0, lr 0.009, BCE & Tversky Loss 1:1 
- Green    Unet++, efficientnet-b4, lr 0.002, Dice & BCE Loss 1:3

Pink Run: 

![image](https://github.com/marioholzheu/MLFinalProject/assets/163416187/0c16d70e-a562-408c-90a0-e92f05b0c340)

![Screenshot 2024-05-19 at 22 23 27](https://github.com/marioholzheu/MLFinalProject/assets/163416187/15d6be84-1599-4398-8599-3457faf914b5)

![Screenshot 2024-05-19 at 22 23 38](https://github.com/marioholzheu/MLFinalProject/assets/163416187/d5310614-d350-4c55-a8ef-95ed50c53626)


Qualitative Results: 
It can be seen in the qualitative comparison that the model predicts many parts quite well, also it normally sees all the classes from the ground truth. But its no perfect in drawing the boundaries, and for smaller parts. In my opinion the model should be trained on more data, therefore we could make use of the folds. and train on all 5 folds, if computing power would be available. 

![Screenshot 2024-05-19 at 22 17 46](https://github.com/marioholzheu/MLFinalProject/assets/163416187/103db344-56a8-4ad1-b452-e261818946e4)

##### Conclusion 
The models would still benefit on more Epochs in my opinion as their are still improving, due to long training time (around 10 hours) with 10 Epochs, I did not have the chance to invest more time.  I would also like to try the Efficientnet-b7 Backbone which had quite some promising results in other competitors, which also due to computational needs would lead to high training time. Additionally with more epochs I would also try different schedulers, and a multimodel approach for classification and then segmentation.I think the UNet++ Model could be the most promising model for the challenge, and improvements are still possible. As can be seen in Qualititve Results, Model predictions are not perfect yet and improvements should be done.

#### Unet++
##### Description of Unet++ 

Architecture: 

![Screenshot 2024-05-19 at 11 56 01](https://github.com/marioholzheu/MLFinalProject/assets/163416187/eea7508e-1cdb-4496-9daa-a87ee42b7ac7)
Source: https://miro.medium.com/v2/resize:fit:2000/1*XmqyKSM3I68GWGJg3V5ZkQ.jpeg

UNet++ improves upon the original UNet architecture for image segmentation by incorporating nested and dense skip connections, which enhance feature propagation and capture multi-scale information. This structure allows for better feature fusion, leading to more accurate segmentation results, particularly in handling fine details and small objects. Additionally, UNet++ offers greater flexibility in network design and improves gradient flow during training, making it more adaptable and effective for various complex and imbalanced datasets.

##### Possible Enhancements
- Backbone / Encoder: Efficientnet with more parameters like B7, resnet200, densenet
- More Epochs: 15 - 25 Epochs
- Multimodel Approach: We should use multi models like some competitors
- 2 Step Approach: Combines Multilevel Approach with Approach to use different model for segementation and classification
- Read through more papers and use promising approaches and best hyperparameters
- Look for additional dataset
- Finsih up the infer file for final predictions on truly unseen data

##### Intersting Papers for the Topic:
- U-Net and UNet++ with and without deep supervision: https://arxiv.org/pdf/1807.10165
- Comparison Skip Connections Redesign: https://arxiv.org/pdf/1912.05074v2
- Language meets Vision Transformer in Medical Image Segmentation: https://arxiv.org/pdf/2206.14718
- UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION https://arxiv.org/pdf/2004.08790v1

### 3.2.2 MODEL - Yaseen 
### Chosen Models: 

#### PSPNet:

Model Architecture:

- Purpose: PSPNet is designed for accurate scene parsing by capturing both local and global context information.
- Architecture:
- Backbone: A CNN (e.g., ResNet) is used to extract a feature map from the input image.
- Pyramid Pooling Module: This module pools the feature map at different grid scales, capturing context at multiple levels (e.g., 1x1, 2x2, 3x3, and 6x6).
- Concatenation and Convolution: The pooled features are concatenated and passed through convolutional layers to merge the context information.
- Upsampling and Final Prediction: The feature map is upsampled to the original image size, producing the final segmentation map.
-  Global and Local Context: The pyramid pooling helps in capturing global context while preserving local details, improving the accuracy of segmentation.

 ![Screenshot 2024-05-14 213934](https://github.com/marioholzheu/MLFinalProject/assets/102143794/a649e293-c883-4893-a007-fd4d4a43056b)

 #### My results:
 ![image](https://github.com/marioholzheu/MLFinalProject/assets/102143794/4d4cebe6-5321-4e04-b64f-659d1ec1e5e6)

 
class CFG:
    seed = 101
    debug = False  # set debug=False for Full Training
    exp_name = '2.5D'
    comment = 'deeplabv3-efficientnet_b1-128x160-ep=15'
    model_name = 'DeepLabV3'
    backbone = 'efficientnet-b1'

    train_bs = 16  
    valid_bs = train_bs*2
    img_size = [160, 192]  
    epochs = 5
    lr = 2e-3  
    scheduler = 'CosineAnnealingLR'  
    min_lr = 1e-6
    T_max = int(30000/train_bs*epochs)+50
    warmup_epochs = 5  
    wd = 1e-5  
    n_accumulate = 32
    #n_accumulate = max(1, 32 // train_bs)  
    n_fold = 5
    folds = [0] 
    num_classes = 3  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
![image](https://github.com/marioholzheu/MLFinalProject/assets/108298847/14f96cff-1674-4fea-8ad5-154a7ce26c72)


![FPN conceptual](https://github.com/marioholzheu/MLFinalProject/assets/102143794/8e508b2f-603e-4293-a634-90efec8b00be)

#### My results: 
![image](https://github.com/marioholzheu/MLFinalProject/assets/102143794/22c65eaa-8e27-488d-b987-db40e2f4248c)

#### LinkNet:

Model Architecture:

 - Purpose: LinkNet is designed for efficient semantic segmentation, balancing speed and accuracy.
 - Architecture:
 - Encoder-Decoder Structure: The network consists of an encoder to capture features and a decoder to reconstruct the segmentation map.
 - Residual Connections: Each stage of the encoder is connected to the corresponding stage of the decoder, which helps in retaining spatial information and gradients flow.
 - Skip Connections: Features from the encoder are directly connected to the decoder stages, helping in better reconstruction of the segmentation map.
 - Efficiency: LinkNet is designed to be lightweight and fast, making it suitable for real-time applications.

![Screenshot 2024-05-16 103610](https://github.com/marioholzheu/MLFinalProject/assets/102143794/07f61fce-76b9-421b-b677-617e9d476f0f)

 - Differences between models:
 - FPN: Focuses on creating multi-scale feature maps with a top-down pathway and lateral connections, enhancing performance for various object sizes.
 - LinkNet: Employs an efficient encoder-decoder structure with residual and skip connections, suitable for real-time segmentation.
 - PSPNet: Utilizes a pyramid pooling module to capture global and local context, achieving high accuracy in scene parsing.

 #### My results:
![image](https://github.com/marioholzheu/MLFinalProject/assets/102143794/0c702c0b-dcf0-4cc0-8f07-fbc69c981176)

In my approach, FPN with resnet performs better : 

![Screenshot 2024-05-19 135645](https://github.com/marioholzheu/MLFinalProject/assets/102143794/23237989-88d2-492c-9b8f-b39ed256e5da)
![Screenshot 2024-05-19 135608](https://github.com/marioholzheu/MLFinalProject/assets/102143794/a3d1c8cf-1bb4-40ed-8870-530dd7b33503)

Training time was around 3.5 hours for both FPN models with EfficientnetB0 and Resnet34. 
##### Resnet architecture:
 - Valid Dice : 0.9108
 - Valid Jaccard :0.8820

 ##### Efficienetnet architecture:
 - Valid Dice : 0.9015
 - Valid Jaccard :0.8727

 Qualitative results in UNet vs Resnet34-FPN vs Efficienet-FPN:
 FPN -resnet34:
![FPnet-resnet34-15epochs](https://github.com/marioholzheu/MLFinalProject/assets/102143794/67e95128-2f04-4f66-a542-65b73130d8cf)

 Unet -Efficientnetb0:
![efficientnetb2-unet](https://github.com/marioholzheu/MLFinalProject/assets/102143794/4383fffa-a618-49b9-b3bf-d22266bb0a41)

 FPN -Efficientnetb2: 
![image](https://github.com/marioholzheu/MLFinalProject/assets/102143794/d71071af-13dc-4eb3-8a84-87ef38575a91)

Further research would involve comparison with existing SOTA approaches and taking into account existing research on this specific task. Some interesting papers for this challenge:
##### FPN efficientnetb0: 
https://www.mdpi.com/2075-4418/13/14/2399

##### PSPnet with multiple backbones:
https://ieeexplore.ieee.org/abstract/document/10328779

### 3.2.3 MODEL - Shiva
Chosen Model: DeepLabV3:


Purpose: DeepLabV3 is tailored for high-precision semantic segmentation tasks, emphasizing both accuracy and computational efficiency.
Architecture:  
![image](https://github.com/marioholzheu/MLFinalProject/assets/108298847/2091b4a7-295b-4097-87e2-a73836844bc8)

**Encoder-Decoder Structure:** Similar to other semantic segmentation models, DeepLabV3 comprises an encoder to extract features and a decoder to refine segmentation details.
**Dilated Convolutions:** DeepLabV3 utilizes dilated convolutions in the encoder to effectively enlarge the receptive field without increasing parameters or computational cost excessively.
**ASPP (Atrous Spatial Pyramid Pooling):** This module in the encoder employs parallel dilated convolutions with different rates to capture multi-scale contextual information.
**Skip Connections:** To preserve spatial details, skip connections are employed, integrating features from different scales into the decoder.
CRF (Conditional Random Fields): DeepLabV3 often incorporates CRF post-processing to refine segmentation boundaries and improve localization accuracy.
**Efficiency:** Despite its sophisticated architecture, DeepLabV3 is engineered for efficiency, achieving remarkable performance while being computationally feasible for various real-world applications.


**Difference between UNet and DeepLab:**
UNet:

Purpose: Specialized for biomedical image segmentation, excelling in fine detail capture and small object handling.
Architecture: Encoder-decoder with skip connections, emphasizing high-resolution feature preservation.
Use Cases: Primarily used in biomedical imaging tasks like cell segmentation and tumor detection.

DeepLab:

Purpose: Designed for general-purpose semantic segmentation tasks, aiming for high accuracy with computational efficiency.
Architecture: Utilizes dilated convolutions and atrous spatial pyramid pooling for multi-scale context capturing, often with an encoder-only structure.
Use Cases: Widely applicable in various domains, including satellite imagery analysis, autonomous driving, and robotics.

**Quantitative results**
Metrics and Parameters set:
Backbone = Resnet 34
Epoch = 5


![image](https://github.com/marioholzheu/MLFinalProject/assets/108298847/4a776ba0-0b5d-4b45-8ff8-ae13e206e1c1)

Metrics and Parameters set:
Backbone = Effiecientnet 01
Epoch = 10

![image](https://github.com/marioholzheu/MLFinalProject/assets/108298847/97a1fa56-4905-424b-affd-49a7f84c2026)

**Qualitative results**


![image](https://github.com/marioholzheu/MLFinalProject/assets/108298847/146ad230-65dc-4be1-8581-984286ac8cdf)
Some of the observations from the results:

High Accuracy: Clear delineation of large bowel, small bowel, and stomach.​
Precision: Minimal misclassification with distinct organ boundaries.​
Robust Generalization: Consistent performance across various abdominal cross-sections.


### 3.2.4 MODEL - Nikhila
Chosen Model: Model Architecture of MA-Net

The Multi-Scale Attention Net (MA-Net) can capture rich contextual dependencies based on the attention mechanism. Two blocks are designed, Position-wise Attention Block (PAB) and Multi-scale Fusion Attention Block (MFAB). The PAB is used to model the feature interdependencies in spatial dimensions, which capture the spatial dependencies between pixels in a global view. In addition, the MFAB is to capture the channel dependencies between any feature maps by fusing high and low-level semantic features. The channel dependencies of high-level and low-level feature maps are fused in a sum manner, which aims to obtain rich Multi-scale semantic information of feature maps by using attention mechanism and improve network performance. 

Model Architecture:
<img width="853" alt="Screenshot 2024-05-16 at 14 42 45" src="https://github.com/marioholzheu/MLFinalProject/assets/75714795/e64702b3-bf3a-4b2d-b17b-dd086c2c2262">

Qualitative Analysis

![image](https://github.com/marioholzheu/MLFinalProject/assets/75714795/a4665641-d851-4a08-a8ff-8bd057d64ad8)

Prediction

![image](https://github.com/marioholzheu/MLFinalProject/assets/75714795/0598ba4b-b055-46a1-871a-182e6fec5713)
<img width="953" alt="manet graphs" src="https://github.com/marioholzheu/MLFinalProject/assets/75714795/89acd5d0-b72b-48df-960f-7a8003d3cb99">

Learnings: I had to optimize the hyperparameters to improve generalization. The use of attention gates in the architecture and multi-scale fusion increases model complexity, potentially affecting training time and memory requirements. The model predicted constant values for valid dice and valid jaccard for higher learning rates, with NaN(not a number) values for train loss and valid loss. 


# 4. Conclusion: 

In the UW-Madison GI Tract Image Segmentation Challenge, UNet++ emerged as the top-performing model in terms of valid dice, surpassing the baseline UNet with its advanced nested and dense skip connections, achieving higher Dice coefficients and Jaccard indices. FPN also showed promising results, coming close to UNet++ in performance. Additionally, DeepLabV3 together with B1 shows really good scores, but the charts indicate there is an issue as it is constantly going up to scores above 0.95 and probably higher if trained longer. Backbones with higher count of parameters, especially EfficientNet-b4, greatly enhanced segmentation accuracy. Further improvements could be achieved with more extensive training epochs and the use of more powerful backbones like EfficientNet-b7. Implementing a multi-model approach and a two-stage segmentation strategy could refine the process further. Expanding the dataset and employing advanced augmentation techniques would improve generalization to unseen data. Incorporating the latest research, such as Vision Transformers and attention mechanisms, could offer substantial performance boosts. Continued innovation in these models will significantly enhance medical imaging technology, leading to better cancer treatment outcomes and more efficient healthcare delivery.
















