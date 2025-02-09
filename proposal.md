
# SOTA Vision Encoders for India and the world

## Background

### Encoders

An encoder is a model that transforms input data (text, image, audio, or video) into a corresponding vector representation within a vector space. This vector is commonly referred to as an "embedding". Ideally, these embeddings are semantically meaningful; inputs that are related to each other should have embeddings that are closer in terms of a chosen distance metric, typically cosine similarity or Euclidean distance. For example, if a text encoder generates embeddings for "food," "restaurant," and "book," we would expect that *d*("food", "restaurant") < *d*("food", "book"), where *d* represents the distance metric.

Encoders are crucial components in modern multimodal Large Language Models (LLMs).  Most such LLMs utilize frozen, pre-trained encoders for each modality, feeding the resulting embeddings to the LLM.  In this context, vision encoders can be considered the "eyes" of an LLM, while audio encoders serve as its "ears". To be very clear, the LLM does not have direct access to the raw image data but only to their embeddings—a testament to the representational power of these embeddings. See a [survey of vision LLMs](https://www.artfintel.com/p/papers-ive-read-this-week-vision) for a broader overview of how vision LLMs use vision encoders.

### CLIP

CLIP ([arxiv](https://arxiv.org/abs/2103.00020) | [blog](https://openai.com/index/clip/)) comprises two encoders: one for text and another for images. These encoders are trained jointly; the model receives (image, caption) pairs, and the objective is to ensure that the text embeddings of captions are close to their corresponding image embeddings, while remaining distant from unrelated image embeddings.

Since CLIP's introduction, various variants have been developed, including an open-source reproduction called [OpenCLIP](https://github.com/mlfoundations/open_clip) and extremely small [MobileCLIP](https://github.com/apple/ml-mobileclip), designed for edge devices.  Currently, [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) is among the most widely used and performant vision encoders.

## Goals

Our objective is to create vision encoders that:

1.  **Achieve state-of-the-art performance** on established global benchmarks.
2.  **Demonstrate significantly improved understanding of Indian culture** compared to existing SOTA vision encoders.
3.  **Address and resolve key limitations** present in current vision encoder designs.

We also intend to incorporate relevant parallel research, such as [Matryoksha Nested Embeddings](https://arxiv.org/abs/2205.13147), into the training of these models.

While achieving SOTA performance on global benchmarks is a well-defined goal, we elaborate further on the aspects of understanding Indian culture and addressing existing limitations.

### Understanding Indian Culture

The original CLIP paper does a very good job at this.  Beyond standard benchmarks like ImageNet and COCO, they perform comprehensive evaluations across diverse use cases, including food classification, gender classification, celebrity identification, OCR, and object counting.

However, equivalent benchmarks for the Indian context are often lacking or don't exist at all, and thus require a concerted effort to bridge this gap.  Substantial effort may also be required to collect training data for these tasks. For instance, datasets used to train existing vision encoders may not have enough data for OCR for Indian scripts.

This gap extends to text encoders as well. It would be worthwhile to explore models like mBERT, IndicBERT as a foundation for creating these text encoders.

It is important to note that relying solely on synthetic data or captions from Gemini/GPT4 may not be sufficient if the underlying encoders and LLMs also lack a nuanced understanding of Indian cultural aspects.

### Addressing Existing Limitations

We shouldn't just create SOTA models. While we are at it, we should fix any issues in current vision encoders. Few things off the top of my head are: multi-resolution support, context-guided embeddings, and task augmentation.

#### Multi-Resolution Support

Current CLIP models operate at a fixed resolution, such as 384x384 pixels. This imposes a significant constraint, as it often requires arbitrary cropping of images, potentially discarding crucial information. For example, cropping the top or bottom of a 3:4 (width:height) instagram's photo, could completely remove a person's face. We should enable support for images of any width and height, similar to modern image generation models. This would also help us on the efficiency front by utilizing the fact that different tasks require different resolutions, for example: object detection can be done on much smaller resolutions than OCR. 

#### Context-Guided Embeddings

It's not really easy to guide clip models without finetuning, as acknowledged in the original paper:

> Finally, we've observed that CLIP's zero-shot classifier can be sensitive to wording or phrasing and sometimes require trial and error "prompt engineering" to perform well.

Consider a scenario with multiple full-portrait images of 100 men wearing shirts.  CLIP's embeddings are inherently detached from the specific downstream task.  The embeddings might be used for facial recognition or to find similar shirts in an e-commerce setting.  Implementing a post-training like phase, *or perhaps even incorporating this guidance during the initial training*, where we can direct the vision encoder based on the context of the downstream task would significantly enhance its utility and potentially its robustness.

#### Task Augmentation

We propose incorporating tasks like dense pose estimation and object detection directly into the training process.  This should enhance the representational robustness of the embeddings by pre-adapting them to these tasks and improving performance on downstream tasks such as object counting.

## Training Cost Estimation

SigLIP achieved a 71% zero-shot accuracy on ImageNet after being trained for only 48 TPUv4 days.

A TPUv4 has 275 TFLOPS of compute, 1.2 TB/s of memory bandwidth, and 32 GB of HBM memory ([source](https://cloud.google.com/tpu/docs/v4)). In contrast, an H100 offers 2,000 TFLOPS (approximately 7x), 3.3 TB/s of bandwidth, and 80 GB of RAM ([source](https://www.nvidia.com/en-in/data-center/h100/)).

Based on these specifications, the training cost would be roughly equivalent to 7 H100 days.  This translates to approximately $350 (7 days * 24 hours/day * $2/H100 hour). Using optimizations like exploring FP8 training and exploring data and efficiency improvements, would help reduce this cost even.

## Evaluation Benchmarks

In addition to standard global benchmarks, we will evaluate our encoder on the following India-specific categories:

| Category                             | Comments & Examples                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Celebrities (Face2Name)              | Influencers, actors, politicians, sports figures, business leaders, and historical figures (e.g., Maharana Pratap) |
| OCR for Indian Languages             | Indian signboards displaying various scripts                                                                    |
| Festivals                            | Holi, Diwali Chhat                                                              |
| Movie Scenes & Cartoons              |                                                  |
| Famous Characters (Movies & Cartoons) | Sardar Khan, Jetha Lal, etc.                                                                               |
| Edibles                              | Food, vegetables, sweets, and snacks. Examples: masala dosa vs benne dosa; Indian-specific vegetables (e.g., patta gobhi, gobhi); and vegetation such as marigolds, peepal trees, and mango trees. |
| Indian Clothing                       | Examples: Bandhej sari vs Banarasi saree                                                                      |
| GeoGuessr (Indian Locations)           | Iconic Indian locations, monuments, food establishments, shops, etc.                                              |
| Indian Gods                          | Hindu gods, regional deities                                                                                 |
| Indian Colors                        | Kesariya, Rani Pink, Peacock Blue                                                                             |
| Architectural Styles                 |                                                                          |

## Getting Started

-   **CLIP on CIFAR-10:**  Train a CLIP-like model using [CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10/). This dataset contains 60,000 32x32 images across 10 classes.
    -   Begin with the [MobileCLIP](https://github.com/apple/ml-mobileclip) architecture, initializing it randomly (without pre-trained weights).
    -   Then, understand, compare with (siglip, etc) & code up the architecture on your own. 
-   **Mini-CLIP:**  Train a CLIP-like model on a subset O(1M) images of the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset.
-   **Evaluate Mini-CLIP:** Evaluate Mini-CLIP on widely used benchmarks like ImageNet and CIFAR-10 zero-shot.

## Relevant Papers & References
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