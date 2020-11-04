package com.octopus.mlp.core.functions

import kotlin.math.pow

class SigmoidalTransfer : TransferFunction  {
    override fun evalute(value: Double): Double {
        return 1 / (1 + Math.E.pow(-value));
    }

    override fun evaluteDerivate(value: Double): Double {
        return (value - value.pow(2));
    }
}