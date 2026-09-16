"""TensorFlow port of the pinned SakanaAI CTM RL vector core.

Source: third_party/ctm_reference (Apache-2.0), revision below.
Ported operations: ClassicControlBackbone, RL synapses, SuperLinear/GLU NLM,
learned pre/post traces, and first-last sliding-window synchronisation.
World-model distributions and task heads deliberately live outside this file.
Dropout is fixed to zero; only the RL-supported first-last selection is exposed.
"""
import math

import numpy as np
import tensorflow as tf

import tools

UPSTREAM_REVISION = '4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466'
REFERENCE_TORCH_VERSION = '2.7.0+cpu'


def glu(x):
    value, gate = tf.split(x, 2, axis=-1)
    return value * tf.sigmoid(gate)


def dense(ins, outs):
    # PyTorch Linear's default U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    limit = ins**-.5
    layer = tf.keras.layers.Dense(outs, dtype='float32',
        kernel_initializer=tf.keras.initializers.RandomUniform(-limit, limit),
        bias_initializer=tf.keras.initializers.RandomUniform(-limit, limit))
    layer.build((None, ins))
    return layer


class TorchLayerNorm(tools.Module):
    """Population variance over the final axis; center before scaling like LN.

    Explicit centering avoids Keras' batch-normalization-style x*scale+offset
    rearrangement amplifying float32 differences across recurrent ticks.
    """
    def __init__(self, width):
        super().__init__()
        self.gamma = tf.Variable(tf.ones([width]), name='weight')
        self.beta = tf.Variable(tf.zeros([width]), name='bias')

    def __call__(self, x):
        centered = x - tf.reduce_mean(x, -1, keepdims=True)
        variance = tf.reduce_mean(tf.square(centered), -1, keepdims=True)
        return centered * tf.math.rsqrt(variance + 1e-5) * self.gamma + self.beta


def norm(width):
    return TorchLayerNorm(width)


class SuperLinear(tools.Module):
    """Exact upstream axis layout, optional history LN, and learned divisor T."""
    def __init__(self, ins, outs, neurons, do_norm):
        super().__init__()
        limit = 1 / math.sqrt(ins + outs)
        self.w1 = tf.Variable(tf.random.uniform([ins, outs, neurons], -limit, limit), name='w1')
        self.b1 = tf.Variable(tf.zeros([1, neurons, outs]), name='b1')
        self.T = tf.Variable([1.], name='T')
        self.layernorm = norm(ins) if do_norm else None

    def __call__(self, x):
        if self.layernorm is not None:
            x = self.layernorm(x)
        out = tf.einsum('BDM,MHD->BDH', x, self.w1) + self.b1
        # All supported NLM layers have out_dims >= 2 (before GLU).
        return out / self.T


class CTMRLCore(tools.Module):
    """forward(raw_input, pre_trace, post_trace) -> sync, traces, tick traces."""
    def __init__(self, input_size, neurons, memory, ticks, d_input, sync_neurons,
                 nlm_hidden, deep_nlms=True, nlm_norm=True, synapse_depth=1):
        super().__init__()
        if not 1 <= sync_neurons <= neurons or min(memory, ticks, input_size, d_input, synapse_depth) < 1:
            raise ValueError('Invalid CTM core dimensions')
        self.neurons, self.memory, self.ticks = neurons, memory, ticks
        self.sync_neurons, self.synapse_depth = sync_neurons, synapse_depth
        self.backbone1, self.backbone2 = dense(input_size, 2*d_input), dense(d_input, 2*d_input)
        self.backbone_norm1, self.backbone_norm2 = norm(d_input), norm(d_input)
        if synapse_depth == 1:
            self.synapse1, self.synapse2 = dense(d_input+neurons, 2*neurons), dense(neurons, 2*neurons)
            self.synapse_norm1, self.synapse_norm2 = norm(neurons), norm(neurons)
        else:
            widths = np.linspace(neurons, 16, synapse_depth).astype(int).tolist()
            self.first_projection, self.first_norm = dense(d_input+neurons, widths[0]), norm(widths[0])
            self.down, self.up, self.skip_norm = [], [], []
            for lo, hi in zip(widths[:-1], widths[1:]):
                self.down.append((dense(lo, hi), norm(hi)))
                self.up.append((dense(hi, lo), norm(lo)))
                self.skip_norm.append(norm(lo))
        self.nlm1 = SuperLinear(memory, 2*nlm_hidden if deep_nlms else 2, neurons, nlm_norm)
        self.nlm2 = SuperLinear(nlm_hidden, 2, neurons, nlm_norm) if deep_nlms else None
        limit = 1 / math.sqrt(neurons + memory)
        self.start_trace = tf.Variable(tf.random.uniform([neurons, memory], -limit, limit), name='start_trace')
        self.start_activated_trace = tf.Variable(tf.random.uniform([neurons, memory], -limit, limit), name='start_activated_trace')
        # RL source uses the LAST n neurons, despite the base class's out-index
        # buffers describing the first n. Follow the actual RL computation.
        left, right = np.triu_indices(sync_neurons)
        self.left = tf.Variable((left + neurons-sync_neurons).astype(np.int32), trainable=False)
        self.right = tf.Variable((right + neurons-sync_neurons).astype(np.int32), trainable=False)
        self.decay_params_out = tf.Variable(tf.zeros([len(left)]), name='decay_params_out')

    @property
    def output_size(self):
        return self.sync_neurons * (self.sync_neurons+1) // 2

    def initial(self, batch):
        shape = [batch, self.neurons, self.memory]
        return tf.broadcast_to(self.start_trace, shape), tf.broadcast_to(self.start_activated_trace, shape)

    def backbone(self, raw):
        x = self.backbone_norm1(glu(self.backbone1(raw)))
        return self.backbone_norm2(glu(self.backbone2(x)))

    def synapses(self, x):
        if self.synapse_depth == 1:
            x = self.synapse_norm1(glu(self.synapse1(x)))
            return self.synapse_norm2(glu(self.synapse2(x)))
        states = [tf.nn.silu(self.first_norm(self.first_projection(x)))]
        for linear, ln in self.down:
            states.append(tf.nn.silu(ln(linear(states[-1]))))
        out = states[-1]
        for index in range(len(self.up)-1, -1, -1):
            linear, ln = self.up[index]
            out = self.skip_norm[index](tf.nn.silu(ln(linear(out))) + states[index])
        return out

    def trace_processor(self, pre):
        out = glu(self.nlm1(pre))
        if self.nlm2 is not None:
            out = glu(self.nlm2(out))
        return tf.squeeze(out, -1)

    def synchronise(self, post):
        product = tf.gather(post, self.left, axis=1) * tf.gather(post, self.right, axis=1)
        ages = tf.cast(tf.range(self.memory-1, -1, -1), tf.float32)
        decay = tf.exp(-tf.clip_by_value(self.decay_params_out, 0., 4.)[:, None] * ages[None])
        return tf.reduce_sum(product * decay[None], -1) / tf.sqrt(tf.reduce_sum(decay, -1))[None]

    def __call__(self, raw, pre, post):
        features = self.backbone(raw)
        pre_ticks, post_ticks = [], []
        for _ in range(self.ticks):
            state = self.synapses(tf.concat([features, post[:, :, -1]], -1))
            pre = tf.concat([pre[:, :, 1:], state[:, :, None]], -1)
            activated = self.trace_processor(pre)
            post = tf.concat([post[:, :, 1:], activated[:, :, None]], -1)
            pre_ticks.append(state)
            post_ticks.append(activated)
        return self.synchronise(post), pre, post, tf.stack(pre_ticks, 1), tf.stack(post_ticks, 1)

    def torch_named_variables(self):
        """Canonical upstream names -> (variable, transpose when loading/exporting)."""
        result = {}
        def linear(prefix, layer):
            result[prefix+'.weight'] = (layer.kernel, True)
            result[prefix+'.bias'] = (layer.bias, False)
        def ln(prefix, layer):
            result[prefix+'.weight'] = (layer.gamma, False)
            result[prefix+'.bias'] = (layer.beta, False)
        linear('backbone.input_projector.1', self.backbone1)
        ln('backbone.input_projector.3', self.backbone_norm1)
        linear('backbone.input_projector.4', self.backbone2)
        ln('backbone.input_projector.6', self.backbone_norm2)
        if self.synapse_depth == 1:
            linear('synapses.1', self.synapse1)
            ln('synapses.3', self.synapse_norm1)
            linear('synapses.4', self.synapse2)
            ln('synapses.6', self.synapse_norm2)
        else:
            linear('synapses.first_projection.0', self.first_projection)
            ln('synapses.first_projection.1', self.first_norm)
            for i, ((down, down_ln), (up, up_ln), skip) in enumerate(zip(self.down, self.up, self.skip_norm)):
                linear(f'synapses.down_projections.{i}.1', down)
                ln(f'synapses.down_projections.{i}.2', down_ln)
                linear(f'synapses.up_projections.{i}.1', up)
                ln(f'synapses.up_projections.{i}.2', up_ln)
                ln(f'synapses.skip_lns.{i}', skip)
        for i, layer in ((0, self.nlm1), (2, self.nlm2)):
            if layer is None:
                continue
            prefix = f'trace_processor.0.{i}'
            for name in ('w1', 'b1', 'T'):
                result[prefix+'.'+name] = (getattr(layer, name), False)
            if layer.layernorm is not None:
                ln(prefix+'.layernorm', layer.layernorm)
        for name in ('start_trace', 'start_activated_trace', 'decay_params_out'):
            result[name] = (getattr(self, name), False)
        return result
