from models.feature_extractor import YOLOFeatureExtractor
import torch

def test_feature_extractor():

    extractor = YOLOFeatureExtractor(
        model_path="pretrained/yolov8n-seg.pt",
        conf_threshold=0.25,
        device="mps"
    )

    images = torch.rand(4, 3, 640, 640)
    features = extractor.extract_features(images)

    print(features['boxes'].shape)
    #print(features['centers'])
    print(features['classes'].shape)
    print(features['num_objects'])
    #print(features['scores'])
    #print(features['masks'])

    return {
        "features": features,
    }

if __name__ == "__main__":
    test_feature_extractor()


