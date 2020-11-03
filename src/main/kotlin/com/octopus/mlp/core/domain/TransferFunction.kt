package com.octopus.mlp.core.domain

interface TransferFunction {
    fun evalute(value: Double): Double
    fun evaluteDerivate(value: Double): Double
}