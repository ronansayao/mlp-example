package com.octopus.mlp.core

import com.octopus.mlp.core.domain.Layer
import com.octopus.mlp.core.domain.TransferFunction
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream


class MultiLayerPerceptron (

) {

    private var fLearningRate = 0.6
    private lateinit var fLayers: Array<Layer?>
    private var fTransferFunction: TransferFunction? = null


    fun MultiLayerPerceptron(layers: IntArray, learningRate: Double, transferFunction: TransferFunction) {
        fLearningRate = learningRate
        fTransferFunction = transferFunction
        fLayers = arrayOfNulls(layers.size)
        for (i in layers.indices) {
            if (i != 0) {
                fLayers[i] = Layer(layers[i], layers[i - 1])
            } else {
                fLayers[i] = Layer(layers[i], 0)
            }
        }
    }

    fun execute(input: DoubleArray): DoubleArray? {
        var i: Int
        var j: Int
        var newValue: Double
        val output = DoubleArray(fLayers[fLayers.size - 1]!!.Length)

        // Put input
        i = 0
        while (i < fLayers[0]!!.Length) {
            fLayers[0]!!.neurons[i]!!.value = input[i]
            i++
        }

        // Execute - hiddens + output
        var k: Int = 1
        while (k < fLayers.size) {
            i = 0
            while (i < fLayers[k]!!.Length) {
                newValue = 0.0
                j = 0
                while (j < fLayers[k - 1]!!.Length) {
                    newValue += fLayers[k]!!.neurons[i]!!.weights.get(j) * fLayers[k - 1]!!.neurons[j]!!.value
                    j++
                }
                newValue += fLayers[k]!!.neurons[i]!!.bias
                fLayers[k]!!.neurons[i]!!.value = fTransferFunction!!.evalute(newValue)
                i++
            }
            k++
        }


        // Get output
        i = 0
        while (i < fLayers[fLayers.size - 1]!!.Length) {
            output[i] = fLayers[fLayers.size - 1]!!.neurons[i]!!.value
            i++
        }
        return output
    }

    fun backPropagateMultiThread(input: DoubleArray?, output: DoubleArray?, nthread: Int): Double {
        return 0.0
    }

    fun backPropagate(input: DoubleArray?, output: DoubleArray): Double {
        val newOutput = execute(input!!)
        var error: Double
        var i: Int
        var j: Int

        i = 0
        while (i < fLayers[fLayers.size - 1]!!.Length) {
            error = output[i] - newOutput!![i]
            fLayers[fLayers.size - 1]!!.neurons[i]!!.delta =
                error * fTransferFunction!!.evaluteDerivate(newOutput[i])
            i++
        }
        var k: Int = fLayers.size - 2
        while (k >= 0) {

            i = 0
            while (i < fLayers[k]!!.Length) {
                error = 0.0
                j = 0
                while (j < fLayers[k + 1]!!.Length) {
                    error += fLayers[k + 1]!!.neurons[j]!!.delta * fLayers[k + 1]!!.neurons[j]!!.weights[i]
                    j++
                }
                fLayers[k]!!.neurons[i]!!.delta =
                    error * fTransferFunction!!.evaluteDerivate(fLayers[k]!!.neurons[i]!!.value)
                i++
            }

            i = 0
            while (i < fLayers[k + 1]!!.Length) {
                j = 0
                while (j < fLayers[k]!!.Length) {
                    fLayers[k + 1]!!.neurons[i]!!.weights[j] += fLearningRate * fLayers[k + 1]!!.neurons[i]!!.delta *
                            fLayers[k]!!.neurons[j]!!.value
                    j++
                }
                fLayers[k + 1]!!.neurons[i]!!.bias += fLearningRate * fLayers[k + 1]!!.neurons[i]!!.delta
                i++
            }
            k--
        }
        error = 0.0
        i = 0
        while (i < output.size) {
            error += Math.abs(newOutput!![i] - output[i])
            i++
        }
        error /= output.size
        return error
    }

    fun save(path: String?): Boolean {
        try {
            val fout = FileOutputStream(path)
            val oos = ObjectOutputStream(fout)
            oos.writeObject(this)
            oos.close()
        } catch (e: Exception) {
            return false
        }
        return true
    }

    fun load(path: String?): MultiLayerPerceptron? {
        return try {
            val net: MultiLayerPerceptron
            val fin = FileInputStream(path)
            val oos = ObjectInputStream(fin)
            net = oos.readObject() as MultiLayerPerceptron
            oos.close()
            net
        } catch (e: java.lang.Exception) {
            null
        }
    }


    fun getInputLayerSize(): Int {
        return fLayers[0]!!.Length
    }


    fun getOutputLayerSize(): Int {
        return fLayers[fLayers.size - 1]!!.Length
    }









}