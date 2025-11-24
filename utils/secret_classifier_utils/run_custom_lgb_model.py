import lightgbm as lgb
import json
import numpy as np
import re
import string
from collections import Counter
from typing import Dict, List, Optional


class SecretClassifier:
    """
    Classifier for detecting real secrets vs placeholders

    Usage:
        classifier = SecretClassifier(
            model_path='secret_classifier_model.txt',
            metadata_path='model_metadata.json'
        )

        result = classifier.classify("sk_test_abc123...")
        print(result['label'])  # 'REAL' or 'PLACEHOLDER' or 'SUSPICIOUS'
    """

    def __init__(self, model_path: str, metadata_path: str):
        """
        Initialize the classifier by loading model and metadata

        Args:
            model_path: Path to the LightGBM model file (.txt)
            metadata_path: Path to the metadata JSON file
        """
        print(f"Loading model from {model_path}...")
        self.model = lgb.Booster(model_file=model_path)

        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata["feature_columns"]
        self.thresholds = self.metadata["thresholds"]

        print(f"✓ Model loaded successfully")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Thresholds: {self.thresholds}")

    def _calculate_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string"""
        if len(s) == 0:
            return 0.0
        counter = Counter(s)
        length = len(s)
        entropy = -sum(
            (count / length) * np.log2(count / length) for count in counter.values()
        )
        return entropy

    def _extract_features(self, value: str) -> Dict[str, float]:
        """
        Extract all features from a string

        Args:
            value: String to extract features from

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Basic stats
        features["length"] = len(value)
        features["n_upper"] = sum(1 for c in value if c.isupper())
        features["n_lower"] = sum(1 for c in value if c.islower())
        features["n_digits"] = sum(1 for c in value if c.isdigit())
        features["n_special"] = sum(1 for c in value if c in string.punctuation)

        # Proportions
        if len(value) > 0:
            features["prop_upper"] = features["n_upper"] / len(value)
            features["prop_lower"] = features["n_lower"] / len(value)
            features["prop_digits"] = features["n_digits"] / len(value)
            features["prop_special"] = features["n_special"] / len(value)
        else:
            features["prop_upper"] = 0
            features["prop_lower"] = 0
            features["prop_digits"] = 0
            features["prop_special"] = 0

        # Entropy features
        features["entropy"] = self._calculate_entropy(value)
        features["entropy_per_char"] = (
            features["entropy"] / len(value) if len(value) > 0 else 0
        )

        # Character variety
        features["unique_chars"] = len(set(value))
        features["char_variety"] = (
            features["unique_chars"] / len(value) if len(value) > 0 else 0
        )

        # Repetition detection
        if len(value) > 0:
            most_common_char, count = Counter(value).most_common(1)[0]
            features["max_char_freq"] = count / len(value)

            # Longest run of same character
            max_run = 1
            current_run = 1
            for i in range(1, len(value)):
                if value[i] == value[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            features["max_run_length"] = max_run
            features["max_run_ratio"] = max_run / len(value)
        else:
            features["max_char_freq"] = 0
            features["max_run_length"] = 0
            features["max_run_ratio"] = 0

        # Pattern detection
        value_lower = value.lower()

        # Placeholder keywords
        placeholder_keywords = [
            "password",
            "changeme",
            "your_",
            "replace",
            "insert",
            "dummy",
            "example",
            "test",
            "api_key",
            "token",
            "secret",
            "placeholder",
            "enter",
            "paste",
            "add_",
            "put_",
            "fill",
            "config",
            "sample",
            "fake",
            "demo",
        ]
        features["has_placeholder_keyword"] = int(
            any(kw in value_lower for kw in placeholder_keywords)
        )

        # Template patterns
        features["has_template_syntax"] = int(
            bool(re.search(r"[{<%$]\w+[}>%}]", value))
        )
        features["has_curly_braces"] = int("{" in value or "}" in value)
        features["has_angle_brackets"] = int("<" in value or ">" in value)
        features["has_dollar_sign"] = int("$" in value)
        features["has_percent_sign"] = int("%" in value)

        # Hex pattern
        features["is_hex"] = int(bool(re.match(r"^[a-fA-F0-9]+$", value)))
        features["is_hex_40"] = int(bool(re.match(r"^[a-fA-F0-9]{40}$", value)))
        features["is_hex_64"] = int(bool(re.match(r"^[a-fA-F0-9]{64}$", value)))
        features["is_hex_128"] = int(bool(re.match(r"^[a-fA-F0-9]{128}$", value)))

        # Base64 pattern
        features["is_base64_like"] = int(bool(re.match(r"^[A-Za-z0-9+/]+=*$", value)))
        features["has_base64_padding"] = int(
            value.endswith("=") or value.endswith("==")
        )

        # JWT pattern (3 parts separated by dots)
        parts = value.split(".")
        features["has_jwt_structure"] = int(
            len(parts) == 3 and all(len(p) > 10 for p in parts)
        )
        features["n_dots"] = value.count(".")

        # Vendor prefixes
        features["has_stripe_prefix"] = int(value.startswith(("sk_", "pk_")))
        features["has_aws_prefix"] = int(value.startswith("AKIA"))
        features["has_github_prefix"] = int(value.startswith(("ghp_", "gho_", "ghs_")))
        features["has_vendor_prefix"] = int(
            features["has_stripe_prefix"]
            or features["has_aws_prefix"]
            or features["has_github_prefix"]
        )

        # Case patterns
        features["is_all_upper"] = int(value.isupper())
        features["is_all_lower"] = int(value.islower())
        features["has_mixed_case"] = int(
            not features["is_all_upper"]
            and not features["is_all_lower"]
            and features["n_upper"] > 0
            and features["n_lower"] > 0
        )

        # Alphabetic only (common in placeholders)
        features["is_alpha_only"] = int(value.isalpha())

        # Consecutive digits
        features["has_consecutive_digits"] = int(bool(re.search(r"\d{4,}", value)))

        # URL encoding
        features["has_url_encoding"] = int(bool(re.search(r"%[0-9A-Fa-f]{2}", value)))

        # UUID pattern
        features["is_uuid"] = int(
            bool(
                re.match(
                    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
                    value_lower,
                )
            )
        )

        # Underscores and hyphens
        features["n_underscores"] = value.count("_")
        features["n_hyphens"] = value.count("-")
        features["prop_underscores"] = (
            features["n_underscores"] / len(value) if len(value) > 0 else 0
        )

        return features

    def classify(self, text: str) -> Dict[str, any]:
        """
        Classify a string as REAL, PLACEHOLDER, or SUSPICIOUS

        Args:
            text: String to classify

        Returns:
            Dictionary containing:
                - label: 'REAL', 'PLACEHOLDER', or 'SUSPICIOUS'
                - probability: Float between 0 and 1 (probability of being REAL)
                - confidence_zone: 'HIGH' or 'MEDIUM'
                - action: Recommended action ('auto_flag', 'review_queue', or 'ignore')
        """
        # Extract features
        features = self._extract_features(text)

        # Create feature vector in the correct order
        X_input = np.array([[features[col] for col in self.feature_cols]])

        # Predict probability
        proba = self.model.predict(X_input)[0]

        # Determine label based on confidence zones
        if proba >= self.thresholds["high_confidence_real"]:
            label = "REAL"
            confidence_zone = "HIGH"
            action = "auto_flag"
        elif proba >= self.thresholds["medium_confidence"]:
            label = "SUSPICIOUS"
            confidence_zone = "MEDIUM"
            action = "review_queue"
        else:
            label = "PLACEHOLDER"
            confidence_zone = "HIGH"
            action = "ignore"

        return {
            "label": label,
            "probability": float(proba),
            "confidence_zone": confidence_zone,
            "action": action,
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple strings efficiently

        Args:
            texts: List of strings to classify

        Returns:
            List of classification results (same format as classify())
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "n_features": len(self.feature_cols),
            "thresholds": self.thresholds,
            "model_version": self.metadata.get("model_version", "unknown"),
            "training_date": self.metadata.get("training_date", "unknown"),
            "test_metrics": self.metadata.get("test_metrics", {}),
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = SecretClassifier(
        model_path="secret_classifier_model.txt", metadata_path="model_metadata.json"
    )

    # Test cases
    test_cases = [
        "sk_test_51HqJxYKZqJxYKZq9mfkzJxYKZqJxYK",
        "password123",
        "AKIAIOSFODNN7EXAMPLE",
        "changeme",
        "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        "YOUR_API_KEY",
        "{API_KEY}",
        "ghp_abc123def456ghi789jkl012mno345pqr678",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ",
        "insert_your_api_key_here",
    ]

    print("\n" + "=" * 100)
    print("CLASSIFICATION RESULTS")
    print("=" * 100)
    print(
        f"\n{'Text':<50} {'Label':<15} {'Probability':<12} {'Confidence':<12} {'Action':<15}"
    )
    print("-" * 100)

    for text in test_cases:
        result = classifier.classify(text)
        text_display = text[:47] + "..." if len(text) > 50 else text
        print(
            f"{text_display:<50} {result['label']:<15} {result['probability']:<12.4f} {result['confidence_zone']:<12} {result['action']:<15}"
        )

    # Batch classification
    print("\n" + "=" * 100)
    print("BATCH CLASSIFICATION")
    print("=" * 100)

    batch_results = classifier.classify_batch(test_cases[:3])
    for text, result in zip(test_cases[:3], batch_results):
        print(f"\n{text}")
        print(f"  → {result}")

    # Model info
    print("\n" + "=" * 100)
    print("MODEL INFORMATION")
    print("=" * 100)
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
