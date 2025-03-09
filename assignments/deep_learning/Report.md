# Assignment 3 Report

## Deep Learning for Image Classification: Performance Comparison & Transfer Learning
1. Introduction
This report will discuss the models used in this assignment and the results obtained when training and testing the models with the Imagenette and CIFAR-10 datasets.

## Basic CNN: Model Architecture & Performance
1. Architecture
 - 3 Convolutional Layers (using ReLU activation and MaxPooling after each)
 - Fully connected (Dense) Layers implemented using Dropout
 - Cross-Entropy Loss is used for the loss function
 - Model uses the Adam optimizer with a learning rate of 0.001
2. Performance  

<table>
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Training Loss</td>
    <td><strong>0.288</strong></td>
  </tr>
  <tr>
    <td>Validation Loss</td>
    <td><strong>1.170<strong></td>
  </tr>
  <tr>
    <td>Test Accuracy</td>
    <td><strong>68.38%</strong></td>
  </tr>
</table>


3. Observations  
 - The model performed pretty well, but could have done better.
 - Dropout of 50% of neurons in the dense layer may have helped, but generalization was never fully optimized.

## ResNet-18 Model Architecture & Performance
1. Architecture
ResNet-18 is a deep residual network with skip connections that allow deeper architectures to train effectively (chatGPT-4o, March 2025).  I used:  
 - Pretrained ResNet-18 from ImageNet
 - Modified fully connected layer for classification
 - Adam optimizer with a learning rate of 0.001  
2. Performance


<table>
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Training Loss</td>
    <td><strong>0.260</strong></td>
  </tr>
  <tr>
    <td>Validation Loss</td>
    <td><strong>0.591</strong></td>
  </tr>
  <tr>
    <td>Test Accuracy</td>
    <td><strong>76.56%</strong></td>
  </tr>
</table>


3. Observations
 - Significant improvement (**+8.18%**) over the Basic CNN. This shows that ResNet-18 is able to extract superior features.
 - Model may still be overfitting and may benefit from regularization

## Regularization
1. **Data Augmentation**  
The data was augmented in the following ways to overcome overfitting issues:  
 - Random Horizontal Flip (50%)
 - Random Rotations (+/- 15 degrees)
 - Color Jitter (Brightness, Contrast, and Saturation)  

2. Performance Comparison  


<table>
  <tr>
    <th>Model</th>
    <th>Validation Accuracy</th>
    <th>Improvement</th>
  </tr>
  <tr>
    <td>ResNet-18 (No Augmentations)</td>
    <td><strong>76.56%</strong></td>
    <td>Baseline</td>
  </tr>
  <tr>
    <td>ResNet-18 (Augmented)</td>
    <td><strong>86.14%</strong></td>
    <td><strong>+9.58%</strong></td>
  </tr>
</table>


3. Observations  
 - Data augmentation had a significant improvement (**9.58%**) on generalization.
 - Model was better able to handle small variations on sample images.  

## Transfer Learning: Fine-Tuning ResNet-18 on CIFAR-10
1. Testing transfer learning
 - Re-train ResNet-18 from scratch on CIFAR-10
 - Fine-tune the pre-trained ResNet-18 (Imagenette weights) on CIFAR-10
2. Performance Comparison  


<table>
  <tr>
    <th>Model</th>
    <th>Validation Accuracy</th>
    <th>Improvement</th>
  </tr>
  <tr>
    <td>ResNet-18 (From Scratch on CIFAR-10)</td>
    <td><strong>76.57%</strong></td>
    <td>Baseline</td>
  </tr>
  <tr>
    <td>ResNet-18 (Fine-Tuned on CIFAR-10, Pretrained on Imagenette)</td>
    <td><strong>82.04%</strong></td>
    <td><strong>+7.14%</strong></td>
  </tr>
  <tr>
    <td>ResNet-18 (Fine-Tuned on CIFAR-10 with Augmentation)</td>
    <td><strong>83.80%</strong></td>
    <td><strong>+8.79%</strong></td>
  </tr>
</table>


3. Observations  
 - Fine-tuning on Imagenette improved CIFAR-10 accuracy by 7.14%
 - Further fine-tuning with augmentation raised accuracy to 83.80%

4. Rationale for improvements  
 - Imagenette pre-training provided useful feature extraction for edges, shapes, textures, etc.  
 - Fine-tuning adapted the model for CIFAR-10 restrictions
 - Augmentation improved the models ability to generalize

## Final Comparison: All Models


<table>
  <tr>
    <th>Model</th>
    <th>Validation Accuracy</th>
    <th>Key Takeaways</th>
  </tr>
  <tr>
    <td>Basic CNN</td>
    <td><strong>68.38%</strong></td>
    <td>Baseline, struggled with feature extraction</td>
  </tr>
  <tr>
    <td>ResNet-18 (No Augmentations)</td>
    <td><strong>76.56%</strong></td>
    <td>Pretrained ImageNet model improved accuracy</td>
  </tr>
  <tr>
    <td>ResNet-18 (Augmented)</td>
    <td><strong>86.14%</strong></td>
    <td>Regularization improved generalization</td>
  </tr>
  <tr>
    <td>ResNet-18 (Fine-Tuned on CIFAR-10, Pretrained on Imagenette)</td>
    <td><strong>82.04%</strong></td>
    <td>Transfer learning boosted performance</td>
  </tr>
  <tr>
    <td>ResNet-18 (Fine-Tuned on CIFAR-10 with Augmentation)</td>
    <td><strong>83.80%</strong></td>
    <td>Improved performance on CIFAR combining fine-tuning with augmentation</td>
  </tr>
</table>


## Conclusion
The power of deep learning models and transfer learning was leveraged to show that Deep Convolutional Neural Networks are powerful tools for image identification.  
ResNet-18 was able to outperform a basic CNN because it is more adept at feature extraction.  
Data augmentation, as a tool, can help prevent overfitting and improve a model's ability to generalize.  
Transfer learning was demonstrated by fine-tuning the training model using Imagenette before training on CIFAR-10 helped improve the model accuracy.
