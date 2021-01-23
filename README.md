# QPWCNet

This repo contains experimental code for running optical flow and related stuff in Tensorflow 2.

Some efforts have been made to enable quantization through `tensorflow_model_optimization`
and deploy through TFLite, though it is still an ongoing development and does not quite work yet.

## Network architecture

TODO(ycho): Update with most up to date arch

![net](img/net.png)

## Sample output

Here's a sample output from a trained model on an excerpt from the Sintel dataset.


Previous Frame:
![prv](img/2021-01-24-prv.png)

Next Frmae
![nxt](img/2021-01-24-prv.png)

Ground Truth Flow visualization:
![ground-truth](img/2021-01-24-flow-rgb-gt.png)

Predicted Flow visualization:
![pred](img/2021-01-24-flow-rgb.png)

## References

Throughout the development of this project I have been inspired by the works from the following repositories:

* [PWCNet-tf2](https://github.com/hellochick/PWCNet-tf2.git)
* [tfoptflow](https://github.com/philferriere/tfoptflow.git)
