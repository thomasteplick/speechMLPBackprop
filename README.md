<h3>Speech Recognition/Classification using a MLP Neural Network with the Back-Propagation Algorithm</h3>
<hr>
This program is a web application written in Go that makes extensive use of the html/template package.
Navigate to the C:\Users\your-name\SpeechMLPBackprop\src\backprop\ directory and issue "go run speechmlp.go" to
start the Multilayer Perceptron Neural Network server. In a web browser enter http://127.0.0.1:8080/speechMLP
in the address bar.  There are two phases of operation:  the training phase and the testing phase.  During the training
phase, examples consisting of wav audio and the desired class are supplied to the network.  The wav audio files are submitted
to Fourier Analysis using the Fast Fourier Transform (FFT) and the Spectral Power Density in magnitude squared is created.
The mean of the PSD is removed and the magnitude is normalized to one to prevent the gradient from saturating and slowing the learning process.
The network itself is a directed graph consisting of an input layer of the images, one or more hidden layers of a given depth, and
an output layer of nodes.  The nodes of the network are connected by weighted links.
The network is fully connected.  This means that every node is connected to its immediately adjacent neighbor node.  The weights are trained
by first propagating the inputs forward, layer by layer, to the output layer of nodes.  The output layer of nodes finds the
difference between the desired and its output and back propagates the errors to the input layer.  The hidden and input layer
weights are assigned “credit” for the errors by using the chain rule of differential calculus.  Each neuron consists of a
linear combiner and an activation function.  This program uses the hyperbolic tangent function to serve as the activation function.
This function is non-linear and differentiable and limits its output to be between -1 and 1.  <b>The purpose of this program is to classify a
speech waveform stored in a wav audio file</b>.
The user selects the MLP training parameters:
<li>Epochs</li>
<li>Learning Rate</li>
<li>Hidden Layers</li>
<li>Layer Depth</li>
<li>FFT Size</li>
<li>FFT Window</li>
<br />
<p>
The <i>Learning Rate</i> is between .01 and .00001.  Each <i>Epoch</i> consists of 32 <i>Training Examples</i>.  
One training example is a Power Spectral Density (mag^2) and the desired class (0, 1,…, 31).  There are 32 audio wav files and therefore 32 classes.
The WAV audio files were produced using Windows Voice Recorder which produced m4a files.  The m4a files were converted to wav files using Audacity.
The conversion sampling rate was 8kHz, PCM 16-bit, mono.  Each wav file is 2-3 seconds long and consists of the author saying a person's name.  The
person's name is the same as the filename.
</p>
<p>
When the <i>Submit</i> button on the MLP Training Parameters form is clicked, the weights in the network are trained
and the Learning Curve (mean-square error (MSE) vs epoch) is graphed.  As can be seen in the screen shots below, there is significant variance over the ensemble,
but it eventually settles down after about 100 epochs. An epoch is the forward and backward propagation of all the 32 training samples.
</p>
<p>
When the <i>Test</i> link is clicked, 32 examples are supplied to the MLP.  It classifies the audio wav files.
The test results are tabulated and the time or frequency domain plots are available for viewing.  Choose the file, the domain,
the FFT size and window if the frequency domain (spectrum) is wanted to be seen.  If fmedia is available in the PATH environment
variable, the wav file is played in your computer's audio device.
It takes some trial-and-error with the MLP Training Parameters to reduce the MSE to zero.  It is possible to a specify a 
more complex MLP than necessary and not get good results.  For example, using more hidden layers, a greater layer depth,
or over training with more examples than necessary may be detrimental to the MLP.  Clicking the <i>Train</i> link starts a new training
phase and the MLP Training Parameters must be entered again.
</p>

<b>Speech Recognition Learning Curve, MSE vs Epoch, One Hidden Layer, 20 nodes/neurons deep, FFT Size 8192, FFT Window Rectangle,</b>
<b>Learning Rate .01, Epochs 100, 32 classes</b>
![image](https://github.com/user-attachments/assets/1a2e6d80-f76c-4651-adb1-2cf04d702d2c)

<b>Speech Recognition Test Results, One Hidden Layer, 20 nodes/neurons deep, FFT Size 8192, FFT Window Rectangle,</b>
<b>Learning Rate .01, Epochs 100, 32 classes, Time Domain (sec)</b>
![image](https://github.com/user-attachments/assets/6a3a5df6-c199-469d-9f48-0e34a75ba004)

<b>Speech Recognition Test Results, One Hidden Layer, 20 nodes/neurons deep, FFT Size 8192, FFT Window Rectangle,</b>
<b>Learning Rate .01, Epochs 100, 32 classes, Frequency Domain (dB/Hz)</b>
![image](https://github.com/user-attachments/assets/440c3ce6-83c2-484f-b21f-ea170c5a19f1)
![image](https://github.com/user-attachments/assets/a4683f49-0975-4f58-8391-f018e43c04d3)

