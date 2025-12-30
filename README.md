# Dorothea AI - Taylor Swift Song Recommendation Based on Your Mood

![Dorothea AI](https://i.imgur.com/rJ7uaji.jpg)

Welcome to Dorothea AI! This is an artificial intelligence project that provides personalized Taylor Swift song recommendations based on your current mood. Using advanced machine learning techniques and audio feature analysis, Dorothea AI creates a sophisticated mood classification system that matches songs to your emotional state.

## ✨ Features

- **Multi-dimensional Mood Analysis**: Uses valence, energy, danceability, acousticness, and tempo to create comprehensive mood vectors
- **Advanced ML Model**: Neural network architecture with configurable layers and early stopping
- **Flexible Filtering**: Filter recommendations by album, era, or release date
- **Similarity-based Ranking**: Uses cosine similarity for precise mood matching
- **Configurable System**: YAML-based configuration with environment variable overrides
- **Comprehensive Testing**: Property-based testing ensures correctness across all inputs
- **Structured Logging**: JSON-formatted logs with TensorBoard integration

## 🏗️ Architecture

Dorothea AI follows a clean architecture pattern with well-separated concerns:

```
dorothea/
├── config/          # Configuration management
├── data/            # Data processing and validation
├── models/          # ML model training and evaluation
├── recommendation/  # Recommendation engine and similarity
├── persistence/     # Data storage and track mapping
├── utils/           # Logging and utilities
└── tests/           # Unit and property-based tests
```

### Core Components

- **ConfigManager**: Centralized YAML configuration with validation
- **DataProcessor**: CSV loading, validation, and feature normalization
- **MoodClassifier**: Multi-dimensional mood classification system
- **ModelTrainer**: Neural network training with TensorBoard logging
- **RecommendationEngine**: Similarity-based song recommendation
- **TrackMapper**: Spotify track metadata management
- **StructuredLogger**: JSON logging with operation context

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thethiago27/dorothea_ia
   cd dorothea_ia
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time setup):
   ```bash
   python main.py train --dataset data-train/taylor_swift_spotify.csv
   ```

4. **Generate recommendations**:
   ```bash
   python main.py recommend --valence 0.8 --energy 0.6 --limit 5
   ```

## 📖 Usage Guide

### Training a Model

Train a new model with the default dataset:
```bash
python main.py train
```

Train with a custom dataset:
```bash
python main.py train --dataset path/to/your/dataset.csv
```

### Generating Recommendations

Basic mood-based recommendation:
```bash
python main.py recommend --valence 0.8 --energy 0.6
```

Advanced recommendation with filters:
```bash
python main.py recommend \
  --valence 0.2 \
  --energy 0.3 \
  --danceability 0.4 \
  --acousticness 0.7 \
  --album "folklore" \
  --limit 3
```

### Mood Parameters

- **valence** (0.0-1.0): Musical positivity/happiness
- **energy** (0.0-1.0): Intensity and power
- **danceability** (0.0-1.0): Rhythm suitability for dancing
- **acousticness** (0.0-1.0): Acoustic vs electronic sound
- **tempo** (BPM): Song tempo (automatically normalized)

### Available Filters

- **--album**: Filter by specific album name
- **--era**: Filter by Taylor Swift era
- **--limit**: Number of recommendations (default: 10)

## ⚙️ Configuration

Dorothea AI uses YAML configuration files for easy customization. The default configuration is located at `dorothea/config/default_config.yaml`.

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.3
  learning_rate: 0.001
  epochs: 150
  batch_size: 32
  early_stopping_patience: 15
```

#### Data Processing
```yaml
data:
  dataset_path: "data-train/taylor_swift_spotify.csv"
  output_path: "data/processed"
  album_blocklist:
    - "reputation Stadium Tour Surprise Song Playlist"
    - "Speak Now World Tour Live"
  mood_thresholds:
    very_low: 0.2
    low: 0.4
    neutral: 0.6
    high: 0.8
```

#### Recommendation Engine
```yaml
recommendation:
  default_limit: 10
  similarity_metric: "cosine"
  min_confidence: 0.5
```

### Environment Variable Overrides

You can override any configuration value using environment variables:
```bash
export DOROTHEA_MODEL_LEARNING_RATE=0.002
export DOROTHEA_DATA_OUTPUT_PATH="/custom/path"
python main.py train
```

## 🧪 Testing

Dorothea AI includes comprehensive testing with both unit tests and property-based tests:

```bash
# Run all tests
python -m pytest dorothea/tests/

# Run property-based tests only
python -m pytest dorothea/tests/properties/

# Run with coverage
python -m pytest --cov=dorothea dorothea/tests/
```

### Property-Based Testing

The system uses Hypothesis for property-based testing to ensure correctness across all possible inputs:

- **Data validation completeness**
- **Scaler round-trip consistency**
- **Mood classification range validation**
- **Recommendation ranking determinism**
- **Configuration validation**

## 📊 API Reference

### RecommendationEngine API

```python
from dorothea.recommendation.engine import RecommendationEngine
from dorothea.recommendation.schemas import RecommendationRequest
from dorothea.data.schemas import MoodVector

# Create mood vector
mood = MoodVector(
    valence=0.8,
    energy=0.6,
    danceability=0.7,
    acousticness=0.3,
    tempo_normalized=0.5
)

# Create recommendation request
request = RecommendationRequest(
    mood_input=mood,
    filters={"album": "folklore"},
    limit=5
)

# Generate recommendations
response = engine.recommend(request)
```

### MoodClassifier API

```python
from dorothea.models.mood_classifier import MoodClassifier

classifier = MoodClassifier(config)

# Classify a song's mood
features = {
    'valence': 0.8,
    'energy': 0.6,
    'danceability': 0.7,
    'acousticness': 0.3,
    'tempo': 120.0
}

classification = classifier.classify(features)
print(f"Mood: {classification.primary_mood}")
print(f"Confidence: {classification.confidence}")
```

## 🛠️ Technologies Used

- **[Python 3.8+](https://python.org)**: Core programming language
- **[TensorFlow 2.x](https://www.tensorflow.org)**: Machine learning framework
- **[scikit-learn](https://scikit-learn.org)**: Data preprocessing and metrics
- **[pandas](https://pandas.pydata.org)**: Data manipulation and analysis
- **[PyYAML](https://pyyaml.org)**: Configuration file parsing
- **[Hypothesis](https://hypothesis.readthedocs.io)**: Property-based testing
- **[pytest](https://pytest.org)**: Testing framework

## 📁 Project Structure

```
dorothea/
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configuration management
│   └── default_config.yaml      # Default configuration
├── data/
│   ├── __init__.py
│   ├── processor.py             # Data processing pipeline
│   ├── validator.py             # Data validation
│   └── schemas.py               # Data schemas (dataclasses)
├── models/
│   ├── __init__.py
│   ├── mood_classifier.py       # Mood classification system
│   ├── trainer.py               # Model training
│   └── evaluator.py             # Model evaluation
├── recommendation/
│   ├── __init__.py
│   ├── engine.py                # Recommendation engine
│   ├── similarity.py            # Similarity calculations
│   └── schemas.py               # Recommendation schemas
├── persistence/
│   ├── __init__.py
│   └── track_mapper.py          # Track metadata mapping
├── utils/
│   ├── __init__.py
│   └── logging.py               # Structured logging
├── tests/
│   ├── unit/                    # Unit tests
│   ├── properties/              # Property-based tests
│   └── integration/             # Integration tests
└── main.py                      # CLI entry point
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality (both unit and property-based)
3. **Follow the existing code style** and architecture patterns
4. **Update documentation** for any API changes
5. **Submit a pull request** with a clear description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Run linting
flake8 dorothea/
pylint dorothea/

# Run type checking
mypy dorothea/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

Dorothea AI is an independent creation and is not officially affiliated with Taylor Swift or her record labels. This project was developed solely for educational and entertainment purposes.

## 🎵 Enjoy!

Discover and rediscover Taylor Swift's songs that harmonize perfectly with your emotions, courtesy of Dorothea AI!
