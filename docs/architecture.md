# Architecture

The models are decoder-only Transformers with eight pre-normalized blocks, width 512, eight causal attention heads, context 512, tied 8,192-token embeddings and learned positional embeddings. Each block first performs

$$
x=x+\operatorname{Attention}(\operatorname{LayerNorm}(x)),\qquad h=\operatorname{LayerNorm}(x).
$$

The second branch is then added to the **unnormalized residual stream**: `y = x + branch(h)`.

## Standard MLP

$$
\Delta= W_{\mathrm{down}}\operatorname{GELU}(W_{\mathrm{up}}h+b_{\mathrm{up}})+b_{\mathrm{down}}.
$$

Its intermediate width is 2048. This is a conventional dense GELU MLP, not a gated/SwiGLU baseline.

## V50 resident field

The routing projection is a bias-free 512→512 matrix, reshaped into four 128D queries. Every query reads all 512 token coordinates. The four routing spaces do **not** restrict the content responses to disjoint output slices.

For routing space s and resident i:

$$
q_s=\operatorname{normalize}((W_qh)_s),\quad p_{s,i}=\operatorname{normalize}(\widetilde p_{s,i}),
$$
$$
d_{s,i}^2=\|q_s\|^2+\|p_{s,i}\|^2-2q_s^Tp_{s,i},\quad r_{s,i}^2=\exp(\rho_{s,i}),
$$
$$
w_{s,i}=G\left[\max\left(0,1-\frac{d_{s,i}^2}{r_{s,i}^2}\right)\right]^2,
\quad\kappa_{s,i}=\frac{w_{s,i}}{\sqrt{\operatorname{mean}_j(w_{s,j}^2)+1/(MC)}}.
$$

There are M=109 residents per space and C=4 content channels per resident. Gates are nonnegative weights, not probabilities, and sum-to-one normalization is not used. The recorded mixed-precision implementation computes the query projection and routing dot products in ambient BF16. Normalization, vector norms, radius/kernel calculations and gate normalization use FP32. Squared distances retain actual vector norms, including near-zero-vector cases, and are clamped at zero.

The full 512D update is

$$
\Delta=\sum_{s=1}^{4}\sum_{i=1}^{109}\kappa_{s,i}(h)
\sum_{c=1}^{4}\operatorname{SiLU}(u_{s,i,c}^Th+b_{s,i,c})v_{s,i,c}.
$$

A resident owns 128 raw position values, one log squared radius, four 512D input vectors, four input biases and four 512D output vectors: 4,229 learned values. These are static learned parameters during inference, not an evolving physical particle simulation. The current token representation already contains prior-token context through causal attention.

The full resident bank contains 436 residents and 1,744 content activations per block. Its nonlinear responses are computed densely; exact zero gates currently do not skip the associated response computation. There is no additional learned projection or normalization after the resident sum and before residual addition.

## Initialization and optimization

V50 uses V15's complete forward function and initialization, except every resident output weight is multiplied by 0.5 **once at construction**. The output weights are fully trainable afterward. The initial squared radius is 2.171848773956299; fixed kernel gain is 6.617806196212754; response initialization gain is 1.0750019319619315. Raw positions start from independent standard normal coordinates and are normalized for routing.

V50 retains the native TangentAdamW position update, sphere retraction and initial-raw-norm gradient-clipping references. Its position learning-rate multiplier is 0.8671654817246768 relative to the shared base schedule. Other parameters use the original AdamW recipe. The MLP uses ordinary AdamW. See the exported trainers for the exact implementation; replacing V50's optimizer with generic AdamW changes the experiment.
