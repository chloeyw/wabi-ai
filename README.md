# wabi-ai
**Predicting depression from audio features of speech...
**
This effort addresses an automated device for detecting depression from acoustic features in speech. The tool is aimed at lowering the barrier of entry in seeking help for potential mental illness (i.e. financial, language, cultural, physical) and supporting medical professionals' diagnoses. This is NOT designed to replace the roles of medical professionals. 

Early detection and treatment of depression are essential in promoting remission, preventing relapse, and reducing the emotional burden of the disease. Current diagnoses are primarily subjective, inconsistent across professionals, and expensive for the individual who may be in dire need of help. Additionally, early signs of depression are difficult to detect and quantify. These early signs have a promising potential to be quantified by machine learning algorithms that could be implemented in a wearable artificial intelligence (AI) or home device.

Automatic Depression Detection (ADD) is a relatively nascent topic that first appeared in 2009. DepressionDetect presents a novel approach focusing on two aspects that receive scant research attention: class imbalance and data representation (feature extraction).

**Table of Contents
Dataset
Acoustic Features of Speech
Segmentation
Feature Extraction
Convolutional Neural Networks
Class Imbalance
Model Architecture
Training the Model
Results
Donate Your Data
Future Directions
For a code walkthrough, see the src folder. 

**
**Dataset
**
All audio recordings and associated depression metrics were provided by the DAIC-WOZ Database, which was compiled by USC's Institute of Creative Technologies and released as part of the 2016 Audio/Visual Emotional Challenge and Workshop (AVEC 2016). The dataset consists of 189 sessions, averaging 16 minutes, between a participant and virtual interviewer called Ellie, controlled by a human interviewer in another room via a "Wizard of Oz" approach. Prior to the interview, each participant completed a psychiatric questionnaire (PHQ-8), from which a binary "truth" classification (depressed, not depressed) was derived.


**Acoustic Features of Speech
**
While some emotion detection research focuses on the semantic content of audio signals in predicting depression, I decided to focus on the prosodic features, which have also been found to be promising predictors of depression. Prosodic features can be generally characterized by a listener as pitch, tone, rhythm, stress, voice quality, articulation, intonation, etc. Encouraging features in research include sentence length and rhythm, intonation, fundamental frequency, and Mel-frequency cepstral coefficients (MFCCs).2

**Segmentation (code)
**
The first step in analyzing a person's prosodic features of speech is segmenting the person's speech from silence, other speakers, and noise. Fortunately, the participants in the DAIC-WOZ study were wearing close proximity microphones in low noise environments, which allowed for fairly complete segmentation in 84% of interviews using pyAudioAnanlysis' segmentation module. When implementing the algorithm in a wearable device, speaker diarization (speaker identification) and background noise removal would require further development for a more robust product. However, in the interest of quickly establishing a minimum viable product, this desired further development was not addressed in the current effort.


**Feature Extraction (code)
**
There are several ways to approach acoustic feature extraction, which is the most critical component to building a successful approach. One method includes extracting short-term and mid-term audio features such as MFCCs, chroma vectors, zero crossing rate, etc. and feeding them as inputs to a Support Vector Machine (SVM) or Random Forest. Since pyAudioAnalysis makes short-term feature extraction fairly streamlined, my first approach to this classification problem involved building short-term feature matrices from 50ms audio segments of the 34 short-term features available from pyAudioAnalysis. Since these features are lower level representations of audio, the concern arises that subtle speech characteristics displayed by depressed individuals would go undetected.

Running a Random Forest on the 34 short-term features yielded an encouraging F1 score of 0.59, with minimal tuning. This approach has been previously employed by others, so I treated this as "baseline" comparative data for which to develop and evaluate a completely new approach involving convolutional neural networks (CNNs) with spectrograms, which I felt could be quite promising and powerful.

CNNs require a visual image. In this effort, speech stimuli is represented via a spectrogram. A spectrogram is a visual representation of sound, displaying the amplitude of the frequency components of a signal over time. Unlike MFCCs and other transformations that represent lower level features of sound, spectrograms maintain a high level of detail (including the noise, which can present challenges to neural network learning).

An example of a spectrogram input to the CNN is shown in Figure 2.

![image](https://github.com/chloeyw/wabi-ai/assets/140864981/223c85b5-1ac4-43e8-8b7c-8157030a0ce4)


**Convolutional Neural Networks
**
Convolutional neural networks (CNNs) are a variation of the better known Multilayer Perceptron (MLP) in which node connections are inspired by the visual cortex. CNNs have proven to be a powerful tool in image recognition, video analysis, and natural language processing. More germane to the current effort, successful applications have also been applied to speech analysis.

Below is a quick primer to CNNs in the context of my project. For readers interested in more information, Stanford's CS231 course is recommended! 

CNNs take images as input. In the case of the spectrogram, I pass (or input) a grayscale representation, with the "grayness" representative of the audio power level at that specific frequency and time. A filter (kernel) is subsequently slid over the spectrogram image and patterns for depressed and non-depressed individuals are learned (based on the aforementioned "truth" dataset).

The CNN begins by learning features like vertical lines, but in subsequent layers, begins to pick up on features like the shape of frequency-time curve (perhaps representative of speaker intonation). Such learned features may provide an elegant and powerful representation of different prosodic features of speech, which in turn are representative of underlying differences between depressed and non-depressed speech.

However, with the highly detailed representations of speech provided in spectrograms, false noise signals (ambient noise, plosives, unsegmented audio from other speakers, etc.) can be inconveniently picked up by the network. One can mitigate this noise with different regularization parameters in the network (pooling layers, L1 loss functions, dropout, etc.), but unless your training data is abundant, it is challenging for the network to distinguish real predictors of depression from the false signal.


**Class Imbalance (code)
**In the current dataset, the number of non-depressed subjects is about four times larger than that of depressed ones, which can introduce a classification "non-depressed" bias. Additional bias can occur due to the considerable range of interview durations from 7-33 minutes because a larger volume of signal from an individual may emphasize some characteristics that are person specific.

In an attempt to address these issues, each of the participant's segmented spectrograms were cropped into 4-second slices. Next, participants were randomly sampled in 50/50 proportion from each class (depressed, not depressed). Then, a fixed number of slices were sampled from each of the selected participants to ensure the CNN has an equal interview duration for each participant. This drastically reduced the training dataset size to 3 hours from the original 35 hours of segmented audio, which was felt adequate for this exploratory analysis.

It should be noted a few different sampling methods were explored to try to increase the size of the training data, and all resulted in highly biased models in which only the "non-depressed" class was predicted. A revised sampling method should be considered as high-priority in future directions (e.g. see interesting sampling method) to increase the training sample size.


**Training the Model**
I created the model using Keras with a Theano backend and trained it on an AWS GPU-optimized EC2 instance.

The model was trained on 40 randomly selected 513x125 (frequency x time bins) audio segments from 31 participants in each category of depression (resulting in 2,480 spectrograms in total). This is representative of just under 3 hours of audio in order to adhere by strict class (depressed, not depressed) and speaker balancing (160 seconds per subject) parameters. The model was trained for 7 epochs, after which it was observed to overfit based on train and validation loss curves.


**Results**
I assessed my model and tuned my hyperparameters based on AUC score and F1 score on a training and validation set. AUC scores are commonly used to evaluate emotion detection models, because precision and recall can be misleading if test sets have unbalanced classes (although they were balanced with this approach).

The test set (which is distinct from the train and validation sets used to develop the model) was composed of 560 spectrograms from 14 participants (40 spectrograms per participant, totaling 160 seconds of audio). Initially, predictions were made on each of the 4-second spectrograms, to explore the extent to which depression can be detected from 4-second audio segments. Ultimately, a majority vote of the 40 spectrogram predictions per participant was utilized to classify the participant as depressed or not depressed.

