# Rats without Hats
### Contributors:
* [Su Timurturkan](https://github.com/sutimurturkan)
* [Justin Cropsey](https://github.com/jcropsey-gatech)
* [David Gordon](https://github.com/DavidCGordon)


### Summary Figure
![Project Infographic](/media/brain.jpg)

## Introduction
We used Sohel Rana's Parkinson's Disease Classification [PDC](https://www.kaggle.com/sohelranaccselab/parkinsons-disease-classification) dataset on Kaggle.

[Parkinson's Disease](https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/symptoms-causes/syc-20376055) (PD) is a progressive neurodegenerative disorder that is part of the Lewy Body Dementias umbrella that also includes [Dementia with Lewy Bodies](https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/symptoms-causes/syc-20352025) (DLB). As the name of the umbrella term implies, the defining characteristic of both disorders is the presence of Lewy bodies (plaques in the brain). While both disorders ultimately result in the same symptoms, the distinguisher between them is whether the tremor (PD) or another symptom appears first (DLB).

Like other diseases that are caused by plaques in the brain (e.g., Alzheimer's disease), biopsy is off limits due to the dangers associated with neurosurgery. While a form of single-photon emission computerized tomography (SPECT) scan called a dopamine-transporter scan (DaTscan) can assist in diagnosis, it is expensive. Our goal is to elaborate a minimally invasive, low-cost solution using speech characteristics and machine learning to aid in diagnosis of PD.

For our project, we will apply the unstructured learning techniques taught in this class to analyze over 750 dimensions of speech characteristics. Afterwards, we will apply structured learning techniques to the annotated datasets to be able to predict whether a patient exhibiting certain patterns of speech characteristics merits further evaluation for PD.

## Methods
The PDC dataset contains 754 different dimensions of data measuring various speech pathologies along with binary gender categorization. Each patient was sampled three times. There are 64 unaffected individuals and 188 affected individuals represented in the data set (756 samples total). No missing data points were observed in the data set. Following import, the data was split apart into various blocks (viz., headers, patient IDs, and X-Y data). The X-Y data (not including the disease classification (Y)) was scaled using Scikit-Learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

    scaler = StandardScaler()
	scaler.fit(xDataFrame)
	xNumPy = scaler.transform(xDataFrame);

The X-Y data set was sorted based on disease classification (Y) and split into unaffected vs affected sets at the boundary. The two sets were subsequently individually split by a common ratio into training vs testing sets, yielding unaffected-training, unaffected-testing, affected-training, and affected-testing set, while ensuring that all samples of a particular patient went into either their respective training or testing set but not both.

### Unsupervised

A k-means elbow analysis in the range of k = \[4,40\] was performed on the entire data set using Yellowbrick's [K-Elbow Visualizer](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html). Higher orders were examined in narrower ranges due to the increased processing time associated with higher orders.

	model = KMeans();
	visualizer = KElbowVisualizer(model, k=(2,40));

	visualizer.fit(xDataFrame)        # Fit the data to the visualizer
	visualizer.show()

A direct k-means analysis with k = 2 (from the elbow method) was performed on the scaled X data set. The number of representatives in each cluster were tabulated to determine which cluster represented which disease status.

<<<<<<< HEAD
A [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) (PCA) was performed with Scikit-Learn's PCA() over the entire X data set. n was chosen to be 170 because that represented 95.1% of the variance observed in the data. Another k-means elbow analysis in the range of k = \[2, 40\] was performed on the PCA transformed data. Again k was chosen to be 2, and a k-means analysis performed.

Under the assumption that PD is a collection of diseases due to heterogeneity in the rates of degeneration across the various neuroanatomical regions in patients, the PCA and subsequent k-means analysis was performed again, but with the PCA only fitted over the unaffected or the affected subsets of X. (Transformation was performed across the entire data set.) The goal was to identify a well-defined cluster of one of the disease states and then define the other state as everything else (i.e., universe - cluster). Due to the high explained-variance ratio when fitting PCA to unaffected data alone, an additional k = 6 k-means analysis was performed.

A 2-component [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) (GMM) was iteratively run over the scaled data set as well as a 170-component PCA fit to the entire data set. Each generated a [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) with the goal being to identify matrices where one of the diagonals is much larger than the other.

Ten-bin [histograms](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) were prepared on a 5-component PCA analysis fitted to the full, unaffected, and affected data sets, respectively.

### Supervised

For a baseline, a Scikit-Learn Gaussian Naive Bayes was performed.

Due to the overlapping nature of the two categories of data, it was decided to go directly to a neural net (NN) for the next efforts. Keras and TensorFlow were selected as the frameworks.

We experimented with various configurations of NNs, starting with a sequential three-layer NN on grouped-by-individual raw data. The layers were dense and consisted of 188, 4, and 1 neuron(s) per layer respectively, each with sigmoid activation.

Oscillations in the accuracy during training were observed and were traced back to unnormalized data. After fixing this with StandardScaler(), we achieved 100% accuracy on the training data after relatively few epochs, indicating overfitting. After reducing the number of epochs, various combinations of number of layers, density of each layer, activations (rectified linear uint (ReLU) and sigmoid), and epochs were explored.

To compensate for bias in the data set due to the overrepresentation of affected individuals, unaffected individuals were oversampled via [Synthetic Minority Oversampling Technique (SMOTE)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html).

A 100-component PCA transform was fitted to unaffected, original-representation data and then fed into the NN. Again, various combinations of layer count, layer density, activations, and epochs were explored.

Adding a validation\_split argument to the fit() was also explored.

A Random Forest was performed using Scikit-Learn. Various combinations of maximum depth, criterion, features, and number of estimators were explored.

## Results

### Unsupervised

Because the data set is unbalanced in the number of affected vs unaffected individuals, a randomly selected sample from the data set has a 74.6% probability of being affected. In the tables presented below, we are searching for methods that have noticeable deviation from probability due to chance alone.

#### Fig. F1 - K-Means elbow analysis over entire scaled data set
![K-Means Elbow Analysis over Raw Data](/docs/raw_kmeans_elbow2_40.png)

#### Table T1 - Partitioning of unaffected vs affected into clusters by K-Means over entire scaled data set
| Unaffected | Affected | Ratio |
|----|----|----|
| 39 | 322 | 0.892 |
| 153 | 242 | 0.613 |

#### Fig. F2 - K-Means elbow analysis over 170-component PCA data fit to entire data set
![K-Means Elbow Analysis over PCA-170 Entire Data](pca170_kmeans_elbow2_40.png)

#### Table T2 - Partitioning of unaffected vs affected into clusters by K-Means over 170-component PCA data fit to entire data set
| Unaffected | Affected | Ratio |
|----|----|----|
| 38 | 320 | 0.894 |
| 154 | 244 | 0.613 |

#### Fig. F3 - K-Means elbow analysis over 170-component PCA data fit to affected data set
![K-Means Elbow Analysis over PCA-170 Affected Data](pca170_diseased_kmeans_elbow2_40.png)

#### Table T3 - Partitioning of unaffected vs affected into clusters by K-Means over 170-component PCA data fit to affected data set
| Unaffected | Affected | Ratio |
|----|----|----|
| 156 | 247 | 0.613 |
| 36 | 317 | 0.898 |

#### Fig. F4 - K-Means elbow analysis over 170-component PCA data fit to unaffected data set
![K-Means Elbow Analysis over PCA-170 Unaffected Data](pca170_healthy_kmeans_elbow2_40.png)

When the 170-component PCA was fitted to unaffected data, a 0.9984 explained variance was achieved.

#### Table T4 - Partitioning of unaffected vs affected into clusters by K-Means over 170-component PCA data fit to unaffected data set
| Unaffected | Affected | Ratio |
|----|----|----|
| 44 | 337 | 0.885 |
| 148 | 227 | 0.605 |

### K-Means Analysis (with k=6) over 170-component PCA fit to Unaffected Data

#### Table T5 - Partitioning of unaffected vs affected into clusters by K-Means (k=6) over 170-component PCA data fit to unaffected data set
| Unaffected | Affected | Ratio |
|----|----|----|
| 78 | 42 | 0.350 |
| 58 | 159 | 0.733 |
| 10 | 162 | 0.942 |
| 19 | 106 | 0.848 |
| 22 | 87 | 0.798 |
| 5 | 8 | 0.615 |

### Figs. F5 - Two-Component GMM on 170-component PCA fitted to the entire data set
![GMM Confusion Matrix 1](gmm_confusionmatrix_1.png)
![GMM Confusion Matrix 2](gmm_confusionmatrix_2.png)
![GMM Confusion Matrix 3](gmm_confusionmatrix_3.png)
![GMM Confusion Matrix 4](gmm_confusionmatrix_4.png)
![GMM Confusion Matrix 5](gmm_confusionmatrix_5.png)

### Figs. F6 - Five-Component PCA Fitted to Full Data Set
![PCA Component 2 vs. PCA Component 1](full_pca5_1_2.png)
![PCA Component 3 vs. PCA Component 1](full_pca5_1_3.png)
![PCA Component 4 vs. PCA Component 1](full_pca5_1_4.png)
![PCA Component 3 vs. PCA Component 2](full_pca5_2_3.png)
![PCA Component 4 vs. PCA Component 2](full_pca5_2_4.png)
![PCA Component 4 vs. PCA Component 3](full_pca5_3_4.png)

### Figs. F7 - Five-Component PCA Fitted to Affected Data Set
![PCA Component 2 vs. PCA Component 1](affected_pca5_1_2.png)
![PCA Component 3 vs. PCA Component 1](affected_pca5_1_3.png)
![PCA Component 4 vs. PCA Component 1](affected_pca5_1_4.png)
![PCA Component 3 vs. PCA Component 2](affected_pca5_2_3.png)
![PCA Component 4 vs. PCA Component 2](affected_pca5_2_4.png)
![PCA Component 4 vs. PCA Component 3](affected_pca5_3_4.png)

### Figs. F8 - Five-Component PCA Fitted to Unaffected Data Set
![PCA Component 2 vs. PCA Component 1](unaffected_pca5_1_2.png)
![PCA Component 3 vs. PCA Component 1](unaffected_pca5_1_3.png)
![PCA Component 4 vs. PCA Component 1](unaffected_pca5_1_4.png)
![PCA Component 3 vs. PCA Component 2](unaffected_pca5_2_3.png)
![PCA Component 4 vs. PCA Component 2](unaffected_pca5_2_4.png)
![PCA Component 4 vs. PCA Component 3](unaffected_pca5_3_4.png)

### Figs. F9 - Ten-Bin Histogram of 5-Component PCA Fitted to Full Data Set
![Histogram of PCA Component 1](hist10_full_component1.png)
![Histogram of PCA Component 2](hist10_full_component2.png)
![Histogram of PCA Component 3](hist10_full_component3.png)
![Histogram of PCA Component 4](hist10_full_component4.png)
![Histogram of PCA Component 5](hist10_full_component5.png)

### Figs. F10 - Ten-Bin Histogram of 5-Component PCA Fitted to Affected Data Set
![Histogram of PCA Component 1](hist10_affected_component1.png)
![Histogram of PCA Component 2](hist10_affected_component2.png)
![Histogram of PCA Component 3](hist10_affected_component3.png)
![Histogram of PCA Component 4](hist10_affected_component4.png)
![Histogram of PCA Component 5](hist10_affected_component5.png)

### Figs. F11 - Ten-Bin Histogram of 5-Component PCA Fitted to Unaffected Data Set
![Histogram of PCA Component 1](hist10_unaffected_component1.png)
![Histogram of PCA Component 2](hist10_unaffected_component2.png)
![Histogram of PCA Component 3](hist10_unaffected_component3.png)
![Histogram of PCA Component 4](hist10_unaffected_component4.png)
![Histogram of PCA Component 5](hist10_unaffected_component5.png)

### Supervised

The Gaussian Naive Bayes achieved an accuracy of approximately 69% on the testing data set.

Despite the numerous combinations of setups, Keras and TensorFlow consistently achieved 72% +/- 2% on testing data.

The Random Forest achieved a weighted average 75% on the recall.

## Discussion

### Unsupervised

The most obvious problems with our data are the presence of numerous outliers, and the amount of overlap of affected and unaffected individuals. This is most salient from the K-Means "elbow" plots (Figures F1, F2, F3, and F4), which resemble nothing like what an elbow plot should.

As noted in the Results, the data set is biased towards affected individuals by almost 3:1. K-means alone was able to do better than random chance by 17.8% and 19.6% unaffected vs affected, respectively.

The 170-component PCA had lackluster performance in improving the categorization of data relative to a simple K-means clustering on scaled data (0.613:0.894 vs 0.613:0.892). Ironically, fitting the 170-component PCA to only the unaffected data improved the amount of explained variance (99.84% vs 95.1%) but had little impact on the categorization (0.605:0.885 vs 0.613:0.892). This evidences the belief that unaffected individuals constitute a cluster whereas affected individuals are outliers.

Table T6 with the results of the 6-cluster K-means analysis over the 170-component PCA data fitted to the unaffected data set provides a counterargument to this belief due to its above average clustering of affected individuals. As noted in the Introduction, PD is a collection of diseases depending on which nuclei are affected and their respective severities.

Gaussian Mixture Modeling provided interesting results: The results were non-probabilistic (i.e., most entries were classified into a single cluster). In other words, it reduced itself to K-mean. Figures F5 shows a subset of the the confusion matrices resulting from GMM. While we did see the some of the desired results of one of the diagonals being much stronger than the other, given the nature of implications, none of them reached a level where we would feel confident enough to use them for this problem.

Figures F6, F7, and F8 are scatter plots of the various PCA components against each other. They serve to ellucidate the problem with the data overlapping. (The opacity was not attenuated with these graphs, and thus, the orange conceals some of the blue points.)

Figures F9, F10, and F11 are 10-bin histograms of 170-component PCA data fitted to the entire, affected, and unaffected data sets, respectively. When fit to the full data set, the primary PCA component exhibits the most distinction between the two disease states. However, when fit to a biased disease state, higher-order components do contribute to the distinction.

Fitting the PCA to the affected state produced some interesting results: The primary component showed more spread compared to the unaffected group, evidencing that PD is a collection of diseases. The secondary component showed a rightward shift in the affected group. The higher-order components do show additional differences, but not as significant as the lower-order components.

Fitting the PCA to the unaffected state also produced some interesting results: There are distinct shifts in the affected group relative to the unaffected group. Components 1 and 5 show a concentrating effect, which is to be expected with a neuromuscular disease like PD and its loss of fluidity of movement (and speech).

DBSCAN was also performed on the data but failed to perform better than random and thus is not included in the Results.

### Supervised

All of the methods we attempted yielded similar results.

Gaussian Naive Bayes, at 69% accuracy, was 5% below the intrinsic bias of the data set (74.6%).

Keras and TensorFlow provided equally unimpressive results despite the various combinations of setups. Even though most setups were able to achieve 100% accuracy, they achieved sub intrinsic bias scores on the testing data.

Preprocessing the data with PCA also yielded lackluster results due to the low explained variance even for 100-component transformation.

To our disappointment, compensating with SMOTE had no effect since it was performed only on the training data and ultimately yielded already represented individuals.

We were initially enthusiastic with adding validation\_split to the NN because it achieved approximately 95% accuracy on both training and validation. Testing on the testing data set, however, achieved average results. We subsequently realized this was because of the triplicate nature of our data set: The validation data set was in effect a subset of the training data set and thus was already known to the NN. This confirmed our suspicion that maintaining all of an individual's samples in the same grouping (viz., training, validation, or testing) was the appropriate way to handle multiple samples from the same individual. In other words, the intraperson variance is smaller than the interperson variance.

A subsequent comparison of predicted vs actual categorization sorted by individual revealed that, while single and double failures do happen, many failures tended to happen in triples as would be expected with triplicate data. The system simply could not accurately classify a constellation of symptoms it had never observed before.

The Random Forest did not realize any better results because it was encountering the same problems as the NN.

What was ultimately realized is that (1) there is a lot of diversity in the manifestation of symptoms with PD, and (2) the data set used was simply too small to capture this diversity.

There are two lights at the end of the tunnel: The first is that this is a very solvable problem with machine learning given the high accuracy achieved on already seen data. It simply needs to have experienced an individual with a similiar manifestation. The second, alternative approach is that it should be possible to pair down the fields to a subset of significant ones and then score how differently a sample is from "normal".

## Future Work

The first and foremost effort needs to be to augment the data set so that it is more representative of the PD community. This should allow a NN to be trained and accurately predict PD.

Due to time limitations we were unable to randomly assign individuals (with their 3 samples) into training, validation, and testing data sets. This presents the real possibility of a sampling error simply because they were always assigned based on their order in the original data set.

A second promising effort is to manually compare mu and sigma between unaffected and affected with the goal of identifying which features are discriminative between them. Then a simple score card could be devised where if an individual was "outside" of "normal" on a minimum number of dimensions, then the person could be referred for further analysis.

## Acknowledgements

We wish to thank Professor Rodrigo Borela and Karan Singh of Georgia Tech for all their assistance with this project.

## References
[Prediction of Parkinson's disease using speech signal with Extreme Learning Machine](https://ieeexplore.ieee.org/abstract/document/7755419?casa_token=1aO88moUx48AAAAA:uT2CWrt38kw_ULeQK_zidk_ZMNRbEiTi9nNxtUOF3BNBoEbGqBD4UvQZ3chF4Od7-JtjG-i6)

[Parkinson’s Disease Diagnosis in Cepstral Domain Using MFCC
and Dimensionality Reduction with SVM Classifier](https://www.hindawi.com/journals/misy/2021/8822069/)

[LSTM Siamese Network for Parkinson’s Disease Detection
from Speech](https://ieeexplore.ieee.org/abstract/document/8969430?casa_token=IgSUTOkHeJoAAAAA:7DZe463IBhOllBG6uAAlPxUdaIbt9q0qRaykNNijhjD-xXcyW3Ks4WBwwozB8DnbAiL2IZF9)

[Collection and Analysis of a Parkinson Speech Dataset With Multiple Types of Sound Recordings](https://www.researchgate.net/publication/260662600_Collection_and_Analysis_of_a_Parkinson_Speech_Dataset_With_Multiple_Types_of_Sound_Recordings)
=======
[Frank Rosenblatt: Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms](https://link.springer.com/chapter/10.1007/978-3-642-70911-1_20)
>>>>>>> 4ad3ba28acf8add37065e0e4399bf81bcc283645
