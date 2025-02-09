# SOTA Vision Encoders for India & the World

## Background
### Encoder 
An encoder is a model that takes some input (text/image/audio/video) and projects it to a corresponding vector in a vector space. This vector is usually called embedding. We also want these embeddings to be meaningful in some sense. Typically -  inputs that are relevant to each other have embeddings that are closer w.r.t some metric, usually cosine similarity or Euclidean distance. For example, if we get the embeddings of "food" "restaurant" and "book" from a text encoder, then we can assume d(food, restaurant) < d(food, book) where d is some distance metric. 

Encoders are very useful in the sense that most modern multimodal LLMs use a frozen pre-trained encoder for each modality, and provide their embedding to the LLM. So, in a way, vision encoders can be considered eyes of an LLM and audio encoders - ears. See a [survey of vision LLMs](https://www.artfintel.com/p/papers-ive-read-this-week-vision) 

### Clip

Clip ([arxiv](https://arxiv.org/abs/2103.00020) | [blog](https://openai.com/index/clip/)) consissts of two encoders, one for text, and another for image. These encoders are then trained jointly, i.e, the model gets (image, caption) pairs and the goal is that the text embedding of captions are closer to corresponding image embeddings, and far from others. To be more clear, the LLM does not have direct access to the image, but only their embeddings - which is a testament to representational power of these embeddings.

Different variants of clip has been trained since then like an open reproduction called [open clip](https://github.com/mlfoundations/open_clip), and extremely small [mobile clip](https://github.com/apple/ml-mobileclip) that can run on edge devices. The most popular and capable vision encoder is [Siglip](https://huggingface.co/google/siglip-so400m-patch14-384)

## Goal 
We want to create vision encoders that
1. **are state-of-the-art** on global benchmarks. 
2. **understand Indian culture** significantly better than the current SOTA vision encoders. 
3. **fix some of the problems** in current vision encoders along the way. 

We should also incorporate parallel research like [Matryoksha nested Embeddings](https://arxiv.org/abs/2205.13147) into training these models. 

SOTA on global benchmarks is a clearly defined goal, so I'd add more on understanding Indian cutlure, and fixing some of the issues in current vision encoders. 

### Understand Indian culture
The original clip paper does a very good job at this. Other than common benchmarks like ImageNet & COCO, they do thorough evaluations across a set of diverse use cases - food classification, gender classification, identifying a celebrity, OCR, counting, etc. 

Counterparts for many of these benchmarks don't exist for Indian context - and thus we need to first fix that gap. A lot of effort may also be needed to collect trainining data for these tasks, for example, OCR for Indian scripts may not be present a lot in the datasets current vision encoders are trained on. 

We also need to fill these gaps for text-encoders as well. It would be worth looking at mBERT for creating these text-encoders. 

Note that synthetic data or captions may not help a lot here, if the encoders & LLMs they use also lack  understanding Indian cultural nuances. 


### Fix some of the problems. 
There are two things that I can think off the top of my head, multi resolution & context guided embeddings, and mask-specific object detection. 
#### Multi Resolution
Current clip models work at fixed resolution, say 384x384. This is a bottleneck, since most of the time, we're not very sure which part of the image to crop - for exmaple cropping top / bottom of a 3:4 (w:h) instagram photo could just crop a person's face altogether. We should be able to remove this bottleneck and support multi-resolution of any wxh, like current image generation models do. 

#### Context Guided Embeddings
It's not really easy to guide clip models without finetuning. The original paper also makes mentions of it. 
> Finally we've observed that CLIP's zero-shot classifier can be sensitive to wording or phrasing and sometimes require trial and error "prompt enngineer to perform well.

So for example, if you have multiple full portrait images of 100 men wearing shirt & pants. CLIP's embeddings are de-attached from the downstream task. For example - you could be generating the embeddings to do facial recognition, or to find the same shirt in an ecommerce setting. Doing a post-training like phase for clip, where we can guide the vision encoder towards w.r.t the context for downstream tasks, will make it more useful, & probably robust as well. 

#### Task Augmentation
We should add tasks like densepose  & object-detection in the training itself. This should make the embedding's representation more robust by pre adapting them to these tasks and improve downstream performance on tasks like counting.


## Training Cost
Siglip was trained on as few as 48 TPUv4 days achieving 71% accuracy zero-shot on ImageNet. 

A TPUv4 has 275 TFLOPS, 1.2TB/s bandwidth, and 32 GB HBM [source](https://cloud.google.com/tpu/docs/v4). On the other hand, an H100 has 2K TFLOPS (7x), 3.3TB/s bandwidth and 80GB RAM [source](https://www.nvidia.com/en-in/data-center/h100/). 

This would estimate the cost to be of 7H100 days, i.e, 7 days x 24 hours x 2$/H100 ~ 350$. We can possibly train it for much cheaper, by figuring out FP8 training & making other data & efficiency improvements. 


## Evaluation Benchmarks
Apart from the global benchmarks, we should also evaluate our encoder on the following India specific categories. 

| Category                             | Comments & Examples                                                                                           |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Celebrities Face2Name                | Influencers, actors, politicians, sportsmen, businessmen, historical figures (e.g., Maharana Pratap)          |
| OCR for Indian Languages             | Indian signboards in different scripts                                                                        |
| Festivals                            |                                                                                                               |
| Movie Scenes & Cartoons              |                                                                                                               |
| Famous Characters from Movies & Cartoons |                                                                                                               |
| Edibles                              | Food, vegetables, sweets & snacks. Examples: masala dosa vs benne dosa. Many vegetables are unique to India, like patta gobhi, gobhi, etc. Vegetation includes marigold flowers, peepal tree, mango tree. |
| Indian Clothes                       | Examples: Bandhej sari vs Banarasi saree                                                                      |
| GeoGuess                             | Iconic famous Indian places, monuments, food joints, shops, etc.                                              |
| Indian Gods                          | Hindu gods, regional gods                                                                                     |
| Indian Colors                        | Kesariya, Rani Pink, Peacock Blue                                                                             |
| Architectural Styles                 |                                                                                                               |

## How to Get Started?
- **CLIP on CIFAR-10** : Train a clip-like model on [CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10/). The dataset contains 60K 32x32 images for 10 classes.
    - Start with [mobile-clip](https://github.com/apple/ml-mobileclip) architecture (randomly initialized, not pretrained). 
    - Then, understand, compare with (siglip, etc) & code up the architecture on your own. 
- **Mini-CLIP** : Train a clip-like model on a subset O(1M) of [LAION-5B](https://laion.ai/blog/laion-5b/) dataset.
- **Evaluate Mini-CLIP** : Evaluate mini-clip on popular benchmarks like ImageNet & CIFAR-10 zero-shot. 

## Relvant Papers & References
1.  **ViT:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Foundation of Vision Transformers)
2.  **ConvNeXt:** [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (Strong convolutional baseline)
3.  **Swin Transformer:** [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (Handles multi-scale features)
4.  **CLIP:** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Contrastive learning for image-text)
5.  **SigLIP:** [SigLIP: Patch-based Sigmoid Loss for Improved Image-Text Retrieval](https://arxiv.org/abs/2303.15343) (Improved contrastive loss)
6. **LLAVA:** [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
7.  **FLAVA:** [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482)
8. **DINO** (Self-Supervised ViT):  [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)  
9.  **Data Efficient Training:** [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
10. **ALIGN:** [Scaling Up Visual and Vision-Language Representation Learning](https://arxiv.org/abs/2102.05918)  
11. **SAM:**  [Segment Anything](https://arxiv.org/abs/2304.02643)
12. **RANet:** [Resolution Adaptive Networks](https://arxiv.org/abs/1905.10817)
13. **DiNAT**: [Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001)
14. **Focal Modulation Networks:** [Focal Modulation Networks](https://arxiv.org/abs/2203.11926)
