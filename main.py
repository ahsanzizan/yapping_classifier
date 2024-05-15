from yapp_classifier import YappClassifier
from utils.logger import ColorfulLogger

# EXAMPLE USAGE OF THE YAPPING CLASSIFIER MODEL

yapp_classifier = YappClassifier(model_path='./models/yapping_classifier_model',
                                 tokenizer_path='./models/tokenizer.pickle',
                                 stemmer_path='./models/stemmer.pickle')

if __name__ == '__main__':
    input_text = input("Your yapp > ")
    confidence = yapp_classifier.classify_yapp(input_text)

    logger = ColorfulLogger.get_logger('logger')
    logger.info(f"Positive Confidence: {confidence['positive'] * 100:.3f}%")
    logger.error(f"Negative Confidence: {confidence['negative'] * 100:.3f}%")

    classification = max(confidence, key=confidence.get)
    logger.debug(f"The yapp is classified as a {classification} yapp")
