package com.octopus.mlp.core.domain

class Neuron(prevLayerSize: Int) {
    public var value: Double = Math.random() / 10000000000000.0
    public var weights: DoubleArray = DoubleArray(prevLayerSize)
    public var bias: Double = Math.random() / 10000000000000.0
    public var delta: Double = Math.random() / 10000000000000.0

    init {
        for (i in weights.indices) weights[i] = Math.random() / 10000000000000.0
    }
}