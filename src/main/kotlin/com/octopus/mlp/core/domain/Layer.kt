package com.octopus.mlp.core.domain

class Layer(var Length: Int, prev: Int) {

    public var neurons: Array<Neuron?> = arrayOfNulls<Neuron>(Length)

    init {
        for (j in neurons.indices) neurons[j] = Neuron(prev)
    }
}