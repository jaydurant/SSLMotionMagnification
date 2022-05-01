# SSLMotionMagnification

Project exploring various self-supervised methods to generate spatial representations that can be utilized in video motion magnification.

This is a Python3 implementation of 3 Supervised Self Learning algorithms. Namely, MOCOv2, BYOL, and SCRL which are modified in order to work with the downstream task and have the appropriate spatial representations transfered to the Supervised Video Motion Magnifciation as described in [Motion Magnification Paper](https://arxiv.org/pdf/1804.02684.pdf). The self-supervised learning algorithms implementations were based on the implementations introduced in each respective publication but changed accordingly in order to allow for increased augmentations and integration into the codebase

## Dataset

Approximately 200,000 frames were generated in order to train the supervised motion magnificaiton model. The scheme to generate these images involves taking natural images from the COCO data set and using them as background and using segmented objects and placing them randomly in the foreground. Movement was randomined and so was direction. In addition, a magnified frame was generated with spepcific interpolation schemes to prevent blurring and noise. This magnified frame served as the ground truth in the supervised approach in [Oh et al.](https://arxiv.org/pdf/1804.02684.pdf).

For the self-supervised methods 20,000 frames were generated in a similar fashion as described above. Each frame is passed through the ssl networks.

## Hypothesis

Initial attemps tried to generate spatial repersentations through image augmentations which pertained to 