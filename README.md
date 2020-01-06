Python 2.7 -- Cuda-8.0 -- TensorFlow 1.4.0 -- Keras 2.1.5

# Uception
## 3D medical image segmentation with CNN
[[`arXiv`](https://arxiv.org/abs/1812.01752)]

### Data

Data comes from the [[`TubeTk`](https://public.kitware.com/Wiki/TubeTK/Data)] data. More specifically, the 100 Healthy Normal MRIs and MRAs from UNC. A subset of this dataset also includes intra-cranial vasculature (centerline + radius), extracted from the MRA images. These models are found in each patient's "Auxillary Data" folder. Using the command [['ComputeTubeFlyThroughImage'](https://github.com/KitwareMedical/ITKTubeTK/tree/master/apps/ComputeTubeFlyThroughImage)] from the TubeTk apps, one can generate a binary mask from the .tre files in the Auxillary Data.

### Abstract

Deep learning has been shown to produce state of the art results in many tasks in biomedical imaging, especially in segmentation. Moreover, segmentation of the cerebrovascular structure from magnetic resonance angiography is a challenging problem because its complex geometry and topology have a large inter-patient variability. Therefore, in this work, we present a convolutional neural network approach for this problem. Particularly, a new network topology inspired by the U-net 3D and by the Inception modules, entitled Uception. In addition, a discussion about the best objective function for sparse data also guided most choices during the project. State of the art models are also implemented for a comparison purpose and final results show that the proposed architecture has the best performance in this particular context.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1dWlJDlavq7Syb8LowOttDJjXk_7epCfg" width="300px" />
  <img src="https://drive.google.com/uc?export=view&id=1-DdA57MAN4xHGQz-8Q_gEdFJIcb997x9" width="300px" />
</div>
<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=12SLD-5mLe6WMneMSGvY2um0o6nwC0TFL" width="700px" />
</div>
