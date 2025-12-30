import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dorothea.config.settings import ConfigManager, AppConfig
from dorothea.utils.logging import StructuredLogger
from dorothea.data.processor import DataProcessor
from dorothea.data.validator import DataValidator
from dorothea.models.mood_classifier import MoodClassifier
from dorothea.models.trainer import ModelTrainer, ModelEvaluator, ModelExporter
from dorothea.recommendation.engine import RecommendationEngine
from dorothea.recommendation.similarity import SimilarityCalculator
from dorothea.recommendation.schemas import RecommendationRequest
from dorothea.persistence.track_mapper import TrackMapper, TrackInfo
from dorothea.data.schemas import MoodVector
class DorotheaApp:
    def __init__(self, config_path: str = "dorothea/config/default_config.yaml"):
        self.config_manager = ConfigManager()
        self.config: Optional[AppConfig] = None
        self.logger: Optional[StructuredLogger] = None
        self.config_path = config_path
    def initialize(self) -> None:
        try:
            self.config = self.config_manager.load(self.config_path)
            self.logger = StructuredLogger(
                "dorothea.main", 
                level=self.config.logging.level
            )
            self.logger.log_config(self.config.__dict__)
            self.logger.info("Dorothea AI initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Dorothea AI: {e}")
            sys.exit(1)
    def train_model(self, dataset_path: Optional[str] = None) -> None:
        with self.logger.operation_context("DorotheaApp", "train_model") as log:
            try:
                validator = DataValidator()
                processor = DataProcessor(self.config.data.__dict__, validator)
                mood_classifier = MoodClassifier(self.config.data.__dict__)
                trainer = ModelTrainer(self.config.model, self.logger)
                evaluator = ModelEvaluator(self.logger)
                exporter = ModelExporter(self.logger)
                log.info("Initialized training components")
                log.info("Starting data processing pipeline")
                processed_dataset = processor.process_full_pipeline(dataset_path)
                log.info("Data processing completed", 
                        features_shape=processed_dataset.features.shape,
                        labels_count=len(processed_dataset.labels),
                        unique_labels=len(set(processed_dataset.labels)))
                scaler_path = os.path.join(self.config.data.output_path, "scaler.pkl")
                processor.save_scaler(scaler_path)
                log.info("Saved scaler for inference", path=scaler_path)
                from sklearn.model_selection import train_test_split
                X_temp, X_test, y_temp, y_test = train_test_split(
                    processed_dataset.features, 
                    processed_dataset.labels,
                    test_size=0.2, 
                    random_state=42,
                    stratify=processed_dataset.labels
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=0.25,
                    random_state=42,
                    stratify=y_temp
                )
                log.info("Data split completed",
                        train_samples=len(X_train),
                        val_samples=len(X_val),
                        test_samples=len(X_test))
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_val_encoded = label_encoder.transform(y_val)
                y_test_encoded = label_encoder.transform(y_test)
                log.info("Starting model training")
                training_result = trainer.train(X_train, y_train_encoded, X_val, y_val_encoded)
                log.info("Model training completed",
                        training_time=training_result.training_time,
                        final_metrics=training_result.final_metrics)
                log.info("Evaluating model on test set")
                evaluation_metrics = evaluator.evaluate(training_result.model, X_test, y_test_encoded)
                log.info("Model evaluation completed",
                        test_loss=evaluation_metrics.loss,
                        test_mae=evaluation_metrics.mae,
                        macro_f1=evaluation_metrics.get_macro_avg_f1())
                model_base_path = os.path.join(self.config.data.output_path, "model")
                os.makedirs(os.path.dirname(model_base_path), exist_ok=True)
                exporter.save_keras(
                    training_result.model, 
                    model_base_path, 
                    self.config.versioning.model_version
                )
                tfjs_path = os.path.join(self.config.data.output_path, "tfjs_model")
                try:
                    exporter.save_tfjs(training_result.model, tfjs_path)
                    log.info("Model exported in TensorFlow.js format", path=tfjs_path)
                except ImportError:
                    log.warning("TensorFlow.js export skipped (tensorflowjs not installed)")
                log.info("Creating track mapping")
                self._create_track_mapping(processed_dataset, mood_classifier)
                log.info("Model training pipeline completed successfully")
            except Exception as e:
                log.error("Model training failed", exc_info=True)
                raise
    def generate_recommendations(self, 
                                mood_input: Dict[str, float],
                                filters: Optional[Dict[str, Any]] = None,
                                limit: int = 10) -> None:
        with self.logger.operation_context("DorotheaApp", "generate_recommendations") as log:
            try:
                model_path = os.path.join(self.config.data.output_path, "model_latest.h5")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Trained model not found at {model_path}")
                exporter = ModelExporter(self.logger)
                model = exporter.load_model(model_path)
                log.info("Loaded trained model", path=model_path)
                track_mapper_path = os.path.join(self.config.data.output_path, "track_mapping.json")
                track_mapper = TrackMapper(track_mapper_path)
                similarity_calculator = SimilarityCalculator()
                recommendation_engine = RecommendationEngine(track_mapper, similarity_calculator)
                mood_vector = MoodVector(
                    valence=mood_input.get('valence', 0.5),
                    energy=mood_input.get('energy', 0.5),
                    danceability=mood_input.get('danceability', 0.5),
                    acousticness=mood_input.get('acousticness', 0.5),
                    tempo_normalized=mood_input.get('tempo_normalized', 0.5)
                )
                request = RecommendationRequest(
                    mood_input=mood_vector,
                    filters=filters,
                    limit=limit
                )
                log.info("Generating recommendations", 
                        mood_input=mood_input,
                        filters=filters,
                        limit=limit)
                response = recommendation_engine.recommend(request)
                log.info("Recommendations generated",
                        num_recommendations=len(response.recommendations),
                        processing_time_ms=response.processing_time_ms,
                        total_candidates=response.total_candidates)
                self._display_recommendations(response)
            except Exception as e:
                log.error("Recommendation generation failed", exc_info=True)
                raise
    def _create_track_mapping(self, processed_dataset, mood_classifier) -> None:
        processor = DataProcessor(self.config.data.__dict__)
        df = processor.load_dataset()
        df_filtered = processor.filter_albums(df)
        tracks = []
        for i, (_, row) in enumerate(df_filtered.iterrows()):
            features = {
                'valence': row.get('valence', 0.5),
                'energy': row.get('energy', 0.5),
                'danceability': row.get('danceability', 0.5),
                'acousticness': row.get('acousticness', 0.5),
                'tempo': row.get('tempo', 120.0)
            }
            mood_classification = mood_classifier.classify(features)
            track_info = TrackInfo(
                internal_id=i,
                spotify_id=row.get('id', f'unknown_{i}'),
                uri=row.get('uri', f'spotify:track:unknown_{i}'),
                name=row.get('name', f'Unknown Track {i}'),
                album=row.get('album', 'Unknown Album'),
                mood_vector=mood_classification.mood_vector
            )
            tracks.append(track_info)
        track_mapper_path = os.path.join(self.config.data.output_path, "track_mapping.json")
        track_mapper = TrackMapper(track_mapper_path)
        track_mapper.create_mapping(tracks)
        self.logger.info("Track mapping created", 
                        num_tracks=len(tracks),
                        path=track_mapper_path)
    def _display_recommendations(self, response) -> None:
        print("\n" + "="*60)
        print("🎵 DOROTHEA AI RECOMMENDATIONS 🎵")
        print("="*60)
        if not response.recommendations:
            print("No recommendations found for the given mood and filters.")
            return
        print(f"\nQuery Mood Vector:")
        print(f"  Valence: {response.query_mood.valence:.2f}")
        print(f"  Energy: {response.query_mood.energy:.2f}")
        print(f"  Danceability: {response.query_mood.danceability:.2f}")
        print(f"  Acousticness: {response.query_mood.acousticness:.2f}")
        print(f"  Tempo (normalized): {response.query_mood.tempo_normalized:.2f}")
        print(f"\nFound {len(response.recommendations)} recommendations from {response.total_candidates} candidates")
        print(f"Processing time: {response.processing_time_ms:.1f}ms\n")
        for i, rec in enumerate(response.recommendations, 1):
            print(f"{i:2d}. {rec.track_name}")
            print(f"     Album: {rec.album}")
            print(f"     Similarity: {rec.similarity_score:.3f} | Confidence: {rec.confidence:.3f}")
            print(f"     URI: {rec.uri}")
            print()
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dorothea AI - Taylor Swift mood-based recommendation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", 
        default="dorothea/config/default_config.yaml",
        help="Path to configuration file"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    train_parser = subparsers.add_parser("train", help="Train the mood classification model")
    train_parser.add_argument(
        "--dataset", 
        help="Path to training dataset CSV file"
    )
    recommend_parser = subparsers.add_parser("recommend", help="Generate song recommendations")
    recommend_parser.add_argument(
        "--valence", 
        type=float, 
        default=0.5,
        help="Valence (happiness) level (0.0-1.0)"
    )
    recommend_parser.add_argument(
        "--energy", 
        type=float, 
        default=0.5,
        help="Energy level (0.0-1.0)"
    )
    recommend_parser.add_argument(
        "--danceability", 
        type=float, 
        default=0.5,
        help="Danceability level (0.0-1.0)"
    )
    recommend_parser.add_argument(
        "--acousticness", 
        type=float, 
        default=0.5,
        help="Acousticness level (0.0-1.0)"
    )
    recommend_parser.add_argument(
        "--tempo", 
        type=float, 
        default=120.0,
        help="Tempo in BPM (will be normalized)"
    )
    recommend_parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Number of recommendations to return"
    )
    recommend_parser.add_argument(
        "--album", 
        help="Filter by specific album"
    )
    recommend_parser.add_argument(
        "--era", 
        help="Filter by Taylor Swift era (e.g., 'folklore', 'red', '1989')"
    )
    return parser
def main():
    parser = create_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    app = DorotheaApp(args.config)
    app.initialize()
    try:
        if args.command == "train":
            app.train_model(args.dataset)
        elif args.command == "recommend":
            mood_input = {
                'valence': args.valence,
                'energy': args.energy,
                'danceability': args.danceability,
                'acousticness': args.acousticness,
                'tempo_normalized': min(1.0, args.tempo / 250.0)
            }
            filters = {}
            if args.album:
                filters['album'] = args.album
            if args.era:
                filters['era'] = args.era
            app.generate_recommendations(
                mood_input=mood_input,
                filters=filters if filters else None,
                limit=args.limit
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()