from yapp_classifier import YappClassifier

# EXAMPLE USAGE OF THE YAPPING CLASSIFIER MODEL

yapp_classifier = YappClassifier(model_path='./models/yapping_classifier_model',
                                 tokenizer_path='./models/tokenizer.pickle',
                                 stemmer_path='./models/stemmer.pickle')

if __name__ == '__main__':
    input_text = input("Your yapp > ")
    confidence = yapp_classifier.classify_yapp(input_text)
    print(
        f"Negative Confidence: {confidence['negative'] * 100:.3f}%\nPositive Confidence: {(confidence['positive']) * 100:.3f}%")
    classification = max(confidence, key=confidence.get)
    print(
        f"The yapp is classified as a {classification} yapp")
