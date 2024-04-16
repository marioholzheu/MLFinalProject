# 1. About our Challenge
## About the project


### The Challenge
UW-Madison GI Tract Image Segmentation:
https://www.kaggle.com/c/uw-madison-gi-tract-image-segmentation

The primary challenge here is to do segmentation of abdominaél medical MRI-Scans. We want to automatically segement anatomical structures like the stomach and intestines including the identification of the precise pixels in the image. We need to classify for the image slices the stomach, large_bowel and small_bowel and submit together with the pixels of the respective organ. 

### The Problem 
In 2019, around 5 million people worldwide were diagnosed with a type of cancer that affects the digestive system. About half of these patients might receive radiation treatment. This treatment involves targeting the cancer with X-ray beams every day for a few weeks, while trying not to harm the stomach and intestines. New technology allows doctors to see exactly where the tumor and intestines are each day, which is important because their positions can change. Normally, doctors have to manually adjust the radiation equipment to make sure the beams hit the tumor and not the other organs, a process that takes a lot of time. This could take so long that what should be a quick treatment might last an hour, making it hard for patients. There's hope that using deep learning technology could speed this up by automatically adjusting the equipment.

![image](https://github.com/marioholzheu/MLFinalProject/assets/163416187/b5ba0dd8-a8eb-4e3c-9588-32f3af73bfe1)

The UW-Madison Carbone Cancer Center, which has been using this technology since 2015, is supporting a project to improve this process. They're providing anonymous MRI scans from their patients for a competition. The challenge is to create a computer model that can identify the stomach and intestines on these scans quickly. If successful, this would help doctors deliver stronger, safer radiation doses more quickly, improving treatment for cancer patients by reducing side effects and helping them control their cancer better.

### The (Sustainability) Impact 
If this challenge is successfull it allows radiation oncologists to safely deliver higher doses of radiation to the tumor while avoiding the stomach and intestinies. This could speed up the treatments a lot, improve the success rate and reducde side effects. Patients would have a better outcome and doctors could focus on more important tasks and even may see more patients with the saved time. By this it would also contribute to several SDGs. 

This challenge adresses several of the SDGs like Good Health and Well-being, Industry, Innovation and Infrastructure and Partnerships for the Goals:
- Good Health and Well-being (SDG 3): This goal aims to ensure healthy lives and promote well-being for all at all ages. By developing a model that can automatically segment the stomach and intestines on MRI scans, the challenge seeks to enhance the effectiveness and safety of cancer treatments. This can lead to better patient outcomes, fewer side effects, more time for other patients and more accessible treatment options.
- Industry, Innovation, and Infrastructure (SDG 9): The challenge encourages innovation in medical technology by using deep learning to improve cancer treatment processes. The development and implementation of such advanced technologies in healthcare can enhance research and development capacity and promote sustainable industrialization. 
- Partnerships for the Goals (SDG 17): The collaboration between UW-Madison Carbone Cancer Center and the broader research community, including data scientists and technologists around the world, exemplifies a partnership that leverages resources, expertise, and technology to achieve a common goal. This cooperation is essential for sharing knowledge, technology, and resources, thereby fostering an inclusive approach to solving complex global health challenges.


### The Data 

### Evaluation Metrics
This competition is evaluated on the mean Dice coefficient and 3D Hausdorff distance. 

- The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by: ![Screenshot 2024-04-16 at 19 38 58](https://github.com/marioholzheu/MLFinalProject/assets/163416187/fff5c072-3d1c-4098-ab92-a370c2cb3558) where X is the predicted set of pixels and Y is the ground truth. The Dice coefficient is defined to be 0 when both X and Y are empty. The leaderboard score is the mean of the Dice coefficients for each image in the test set.

- Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (In this competition, the slice depth for all scans is set to 1.) The scipy code for Hausdorff is linked. The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

#### Existing Data and Files

#### Data Analysis


# 2. Existing Solutions 
## Overview
Most Used Models
•	Unet: (2D and 2.5D Model) 
•	Monai (3D Model) 
•	DeepLabV3+
•	DeeplabV3

-> Different Approaches: 
•	2D
•	2.5D
•	3D

## Explanation for different Code Solutions

### Train 2.5D U_Net https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-train-pytorch - public score 0.862 + infer

### UNET - well explained: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch - public score 0.842 + infer 


### 3rd Place Winner - not well explained https://www.kaggle.com/code/hesene/3rd-place-winning-solution - public score 0.89 (infer)


### ANOTHER MODEL

### ANOTHER MODEL

## Conclusion
Data Processing, Data Analysis and Data Visualization well explained: 
•	Just for understanding the Project better - with Tensorflow:  https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda 


# 3. Our Code 
## About the Code Base - What changes we had to do?  


## What did we change and try? 


## What other possible solutions could be tried? 





LINK to Styling Syntax:
https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax











Yaseen:
Drawbacks from current model. 
Ambiguity for hyperparameters, not specified as model uses different pretrained models. To make changes in that model we will have to delve deeper into 10 or so pretrained models with pre initialized weights and hyperparameters. 
Code difficult to implement if wamtedto make any changes. 


