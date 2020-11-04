package com.octopus.mlp.core.functions

interface TransferFunction {
    fun evalute(value: Double): Double
    fun evaluteDerivate(value: Double): Double
}