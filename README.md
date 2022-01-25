# Rats with Hats
### Contributors:
* [Su Timurturkan](https://github.com/sutimurturkan)
* [Justin Cropsey](https://github.com/jcropsey-gatech)
* [David Gordon](https://github.com/DavidCGordon)


### Summary Figure
![Project Infographic](/media/brain.jpg)

### Introduction
We used Sohel Rana's Parkinson's Disease Classification (PDC) [Visual Coding](https://www.kaggle.com/sohelranaccselab/parkinsons-disease-classification)dataset on Kaggle.

Parkinson's Disease (PD) is a progressive neurodegenerative disorder that is part of the Lewy Body Dementias umbrella that also includes Dementia with Lewy Bodies (DLB). As the name of the umbrella term implies, the defining characteristic of both disorders is the presence of Lewy bodies (plaques in the brain). While both disorders ultimately result in the same symptoms, the distinguisher between them is whether the tremor (PD) or another symptom appears first (DLB).

Like other diseases that are caused by plaques in the brain (e.g., Alzheimer's disease), biopsy is off limits due to the dangers associated with neurosurgery. While a form of single-photon emission computerized tomography (SPECT) scan called a dopamine-transporter scan (DaTscan) can assist in diagnosis, it is expensive. Our goal is to elaborate a minimally invasive, low-cost solution using speech characteristics and machine learning to aid in diagnosis of PD.

For our project, we will apply the unstructured learning techniques taught in this class to analyze over 750 dimensions of speech characteristics. Afterwards, we will apply structured learning techniques to the annotated datasets to be able to predict whether a patient exhibiting certain patterns of speech characteristics merits further evaluation for PD.


### Methods
The PDC dataset contains 754 different dimensions of data measuring various speech pathologies along with binary gender categorization. Each patient was sampled three times. There are 64 unaffected individuals and 188 affected individuals represented in the data set (756 samples total). No missing data points were observed in the data set. Following import, the data was split apart into various blocks (viz., headers, patient IDs, and X-Y data). The X-Y data (not including the disease classification (Y)) was scaled using Scikit-Learn's StandardScaler.

The X-Y data set was sorted based on disease classification (Y) and split into unaffected vs affected sets at the boundary. The two sets were subsequently individually split by a common ratio into training vs testing sets, yielding unaffected-training, unaffected-testing, affected-training, and affected-testing set, while ensuring that all samples of a particular patient went into either their respective training or testing set but not both.

### Results
Our experiment will have four parts to its results. The first and second, our mid-term "exams" for success, are whether the data is clusterable and whether the clustering is generalizable between participants. The third and fourth parts extend these to determine whether visual stimuli can be predicted from brain readings and whether this is generalizable among participants. Success is defined as correctly predicting the visual stimulus with a probability greater than chance alone.

### Discussion
Although a brain reading model raises many ethical questions, it also has great potential in furthering healthcare technology, particularly for people suffering from neurodenerative(e.g., Lou Gehrig's disease) or neuro-muscular (e.g., myasthenia gravis) disorders where the brain itself remains intact, but the suffer's ability to interact with the outside world is significantly impaired. This technology could be extended to provide them a means to communicate with the outside world after their motor abilities are lost.

### References
[Allen Brain Map](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels)

[Machine learning for neural decoding](https://arxiv.org/ftp/arxiv/papers/1708/1708.00909.pdf)

[Real-Time Decoding of Nonstationary Neural Activity in Motor Cortex](https://ieeexplore.ieee.org/document/4483654)

[Frank Rosenblatt: Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms](https://link.springer.com/chapter/10.1007/978-3-642-70911-1_20)