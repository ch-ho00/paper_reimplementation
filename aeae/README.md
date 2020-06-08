# Affine Equivariant Autoencoder (AEAE)


## Aim of model

The objective consists of the self-reconstruction of the original example and affine transformed example, and the approximation of the affine transformation function, wherethe reconstruction makes the encoder a valid feature extractor and the approximation encourages equivariance.

## Detail of Model

#### Structure of model
<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/img/1.PNG">
Note that the autoencoders at top and bottom share the same parameters w and u. There are three data flows denoted by blue, green, and red dashed lines, making the encoder fw a valid feature extractor for clean example x, affine transformed example Tσ(xi), and equivariant to affine transformations, respectively. In the red flow, the affine parameter σ ∈ R is scaled by element and then added to the last five elements of the feature vector fw(x).

#### Loss function of model
<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/img/2.PNG">
Here Tσ implies the ground truth affine transformation between the two inputs of the model while t(σ) represents predicted factor in which leads to affine transformation during decoding process.

To look at t(σ) in more detail, it has the form of 

<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/img/3.PNG">

where σi's represent the parameters of the affine transformation. 

## Result

|Epoch|Result 1|Result 2|
|---|---|---|
|10|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_100_10.PNG">|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_10_10.PNG">|
|100|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_100_100.PNG">|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_10_100.PNG">|
|150|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_100_200.PNG">|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_10_200.PNG">|
|20w0|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_100_290.PNG">|<img src="https://github.com/chanhopark00/paper_reimplementation/blob/master/aeae/result/result_10_290.PNG">|
