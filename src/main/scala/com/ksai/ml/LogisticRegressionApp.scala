package com.ksai.ml

import ksai.core.classification.LogisticRegression
import ksai.data.parser.ARFFParser

object LogisticRegressionApp extends App{


  private val arffFile = ARFFParser.parse("src/main/resources/iris.arff")

  val trainingInstances: Array[Array[Double]] = arffFile.data.toArray
  val responses: Array[Int] = arffFile.getNumericTargets.toArray

  val logisticRegression = LogisticRegression(trainingInstances, responses)
  val output = trainingInstances.zip(responses).foldLeft(0){
    case (error, (instance, expectedOutput)) if logisticRegression.predict(instance) == expectedOutput=> error
    case (error, _) => error + 1
  }

  println(s"we found error rate: ${(output.toDouble /responses.length) * 100} with error count:  $output")
}
