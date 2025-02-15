Refer to [SOTA Vision Encoders for India & the world](https://github.com/kalpalabs/docs/blob/main/sota_vision_encoders.md) for a rough overview. 

# Starter Projects
Create a directory `starterprojects/{username}/{project_name}` in this repository, and add your work there. This will help new-comers as a reference when they're stuck. 


-   **CLIP on CIFAR-10:**  Train a CLIP-like model using [CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10/). This dataset contains 60,000 32x32 images across 10 classes.
    -   Begin with the [MobileCLIP](https://github.com/apple/ml-mobileclip) architecture, initializing it randomly (without pre-trained weights).
    -   Then, understand, compare with (siglip, etc) & code up the architecture on your own. 
-   **Mini-CLIP:**  Train a CLIP-like model on a subset O(1M) images of the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset.
-   **Evaluate Mini-CLIP:** Evaluate Mini-CLIP on widely used benchmarks like ImageNet and CIFAR-10 zero-shot.