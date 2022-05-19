# DDSP:DIFFERENTIABLE DIGITAL SIGNAL PROCESSING

## DDSP Components
### Spectral modeling synthesis
### Harmonic oscillator / additive synthesizer
Sinusoidal oscillator. A bank of oscillators that outputs a signal $x(n)$ over discrete time steps, $n$, can be expreesed as
$$x(n) = \sum_{k=1}^{K}A_{k}(n)sin(\phi_{k}(n))$$
$A_k(n)$ is the time varying amplitude of the k-th sinusoidal component and $\phi_k(n)$ is its instantaneous phase. The phase $\phi_k(n)$ is obtained by integrating the instantaneous frequency $f_k(n)$
$$\phi_k(n) = 2\pi\sum_{m=0}^nf_{k}(m)+\phi_{0,k}$$
$\phi_{0,k}$ is the initial phase that can be randomized, fixed or learned
m, n是time step; k,K是正弦波组分。output由各个组分的输出乘振幅获得，每个组分的当下相位由频率加初始相位获得。

对于Harmonic oscillator，各个组分频率都是基频倍数，即$f_k(n) = kf_0(n)$。所以harmonic oscillator的output完全由time varying基频$f_0(n)$和harmonic amplitudes$A_k(n)$决定。振幅可以被分解为：
$$A_k(n) = A(n)c_k(n)$$
一个全局振幅$A(n)$，控制==响度==和能够决定频谱变化的==harmonics中的归一化分布c(n)==。满足$\sum_{k=0}^Kc_k(n) = 1$和$c_k(n)>=0$。 本文使用modified sigmoid nonlinearity限制amplitudes和harmonic distribution components为正数。











### Envelopes
oscillator formulation requires ==time-varying amplitudes and frequencies at the audio sample rate,== 但neural networks ==operate at a slower frame rate.== 本文使用bilinear interpolation进行即时frequency upsampling.但是amplitudes and harmonic distributions of the additive synthesizer ==required smoothing== to prevent artifacts. 可以使用a smoothed amplitude envelope by ==adding overlapping Hamming windows== at the center of each frame and scaled by the amplitude

需要上采样->需要smoothing防止人工痕迹->使用smoothing amplitude envelope by adding overlapping Hamming windows
### Filter design: frequency sampling method
standard convolutional layers = LTI-FIR filters
为了保证interpretability&防止phase distortion, 使用frequency sampling method to convert network outputs into ==impulse responses of linear-phase filters.==

neural network to predict the frequency-domain transfer functions of a FIR filter for every output frame. 网络会输出每一帧的transfer function $H_l$. 把 $H_l$ 当作对应FIR filter的transfer function, 从而实现一个time-varying FIR filter.

使用方法是把输入audio分成non-overlapping frames $x_l$, 在==傅里叶域Fourier domain==进行frame-wise卷积$Y_l = H_lX_l$ ,$X_l = DFT(x_l)$, $Y_l = DFT(y_l)$是output, frame-wise filtered audio by $y_l = IDFT(Y_l)$, overlap-add the resulting frames

不把网络输出直接作为transfer function,要在其上加一个window function计算得到$H_l$. 

### Filtered noise / subtractive synthesizer
Natural sounds = harmonic + stochastic
Harmonic plus Noise model captures this by ==combining== the output of an ==additive synthesizer== with a stream of ==filtered noise==. A ==differentiable filtered noise synthesizer== by applying ==LTV-FIR filter== to a stream of ==uniform noise $Y_l = H_lN_l$ where $N_l$  is IDFT of uniform noise in domain $[-1, 1]$.==

### Reverb: Long impulse response
In reverb modeling part, we gain interpretability by explicitly factorizing the room acoustics into a post-synthesis convolution step. 但是room impulse response很长,可能有几秒钟,导致计算复杂度过高. 因此本文采用==频域frequency domain==乘法而不是傅里叶域乘法进行卷积.

## Experiments
tested 2 ddsp autoencoder variants - supervised and unsupervised. 
supervised model is conditioned on 基频 and loudness features extracted from audio.
unsupervised model learns 基频jointly with the rest of the network

### DDSP autoencoder
ddsp 不限制生成模型(GAN, VAE, Flow等等). 图像领域中convolutional layer+autoencdoer > 全连autoencoder, ddsp也可以大大增强autoencoder的性能.

encoder network$f_{enc}()$ maps input x to a latent representation $z = f_{enc}(x)$ and a decoder network $f_{dec}()$  reconstruct the input $x = f_{dec}(z)$ . encoder把输入map到潜在表示中, decoder使用参数重建输入.

#### Encoder
- Supervised: loudness $l(t)$ is extracted directly from the audio, a pretrained CREPE model with fixed weights is used as an f(t) encoder to extract 基频. optional encoder 提取time-varying latent encoding z(t) of the residual information. z(t) encoder, 首先提取MFCC系数, 然后用单层GRU transform.
- Unsupervised: 使用Resnet提取mel-scaled log spectrogram中的基频f(t),然后和网络剩下的部分协同训练

#### Decoder
                        decoder
(f(t), l(t), z(t)) -------> parameters(additive & filtered noise synth) -> generate audio
reconstruction loss between synthesized and original audio is minimized.

==unique point==: latent f(t) is fed directly to the additive synth as it has ==structural meaning== for the synth




### Datasets
#### Multi-scale spectral loss

## Model details
### Encoders
- f-encoder: 
use a ==pretrained CREPE pitch detector== as the f-encoder to extract ==ground truth fundamental frequencies(F0)== from the audio. We used the large variant of CREPE
1. supervised autoencoder, fixed the weights of the f-encoder like [FAFNAS]
2. unsupervised autoencoder, jointly learn the weights of a ==resnet== model fed log mel spectrograms of the audio

- l-encoder: 
use identical computational steps to extract loudness as [FAFNAS]. An A-weighting of the power spectrum, which puts greater emphasis on higher frequencies, followed by log scaling. The vector is then centered according to the mean and standard deviation of the dataset.

- z-encoder: 
first calculate MFCC's frin the audio.
use the first 30 MFCCs that correspond to a smoothed spectral envelope.
MFCCs are then passed through a normalization layer and a 512-unit GRU
GRU output fed to 512-unit linear layer to obtain $z(t)$.
The $z$ embedding reported in this model has 16 dimensions across 250 time-steps



### Decoder
the decoder's input is the latent tuple $(f(t), l(t), z(t))$ . Its ==output are parameters required by the synthesizers==. Harmonic synthesizer and filtered noise synthesizer setup, the decoder ==outputs $\alpha(t)$ (amplitudes of the harmonics)== for the harmonic synth(f(t) is fed directly from the latent), and ==H(transfer function of the FIR filter)  for the filtered noise synthesizer==.

a "shared-bottom" architecture, computes a shared embedding from the latent tuple., and then have one head for each of the $(\alpha(t), H)$ outputs.
H合成器和FN合成器的参数使用shared-bottom架构,输入相同架构相同,最后得到两个参数输出。

Apply separate MLPs to each of the latent tuple input. The outputs of the MLPs are concatenated and passed to a 512 unit GRU. We concatenate the GRU outputs with the outputs of the $f(t)$ and $l(t)$ MLPs and pass it through a final MLP and Linear layer to get the decoder outputs.
架构如图。

MLP是3层， 每层Dense+layer norm+RELU

