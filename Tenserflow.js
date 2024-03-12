import React, {useState} from 'react';
import {View, Text, Button, Image} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

const ImageClassifier = () => {
  const [predictions, setPredictions] = useState([]);

  const classifyImage = async () => {
    const image = Image.resolveAssetSource(require('./assets/cat.jpg'));
    const model = await mobilenet.load();
    const tensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([224, 224])
      .toFloat();
    const predictions = await model.classify(tensor);
    setPredictions(predictions);
  };

  return (
    <View>
      <Image
        source={require('./assets/cat.jpg')}
        style={{width: 200, height: 200}}
      />
      <Button title="Classify Image" onPress={classifyImage} />
      {predictions.map((prediction, index) => (
        <Text key={index}>
          {prediction.className}: {prediction.probability.toFixed(3)}
        </Text>
      ))}
    </View>
  );
};

export default ImageClassifier;
