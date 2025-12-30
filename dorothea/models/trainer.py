"""
Model training components for Dorothea AI.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report
from ..config.settings import ModelConfig
from ..utils.logging import StructuredLogger

@dataclass
class TrainingResult:
    """Result of model training containing model and metrics."""
    model: tf.keras.Model
    history: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    training_time: float
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best validation loss."""
        if 'val_loss' in self.history:
            return int(np.argmin(self.history['val_loss']))
        elif 'loss' in self.history:
            return int(np.argmin(self.history['loss']))
        else:
            return len(self.history.get('loss', [])) - 1
@dataclass
class EvaluationMetrics:
    """Metrics from model evaluation."""
    loss: float = 0.0
    mae: float = 0.0
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    accuracy: Optional[float] = None
    classification_report: Optional[str] = None
    def get_macro_avg_f1(self) -> float:
        if not self.f1_score:
            return 0.0
        return np.mean(list(self.f1_score.values()))
    def get_weighted_avg_f1(self) -> float:
        if 'weighted avg' in self.f1_score:
            return self.f1_score['weighted avg']
        return self.get_macro_avg_f1()
class ModelTrainer:
    """Handles neural network model training with TensorFlow."""
    
    def __init__(self, config: ModelConfig, logger: Optional[StructuredLogger] = None):
        """Initialize the model trainer.
        
        Args:
            config: Model configuration
            logger: Structured logger instance
        """
        self.config = config
        self.logger = logger or StructuredLogger(__name__)
    def build_model(self, input_shape: tuple, output_shape: int = 1) -> tf.keras.Model:
        """Build a neural network model with the configured architecture.
        
        Args:
            input_shape: Shape of input features (excluding batch dimension)
            output_shape: Number of output neurons (1 for regression, n for classification)
            
        Returns:
            Compiled TensorFlow model
        """
        with self.logger.operation_context("ModelTrainer", "build_model") as log:
            log.info("Building model architecture", 
                    input_shape=input_shape, 
                    output_shape=output_shape,
                    hidden_layers=self.config.hidden_layers,
                    dropout_rate=self.config.dropout_rate)
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=input_shape))
            for i, units in enumerate(self.config.hidden_layers):
                model.add(tf.keras.layers.Dense(
                    units, 
                    activation='relu',
                    name=f'hidden_{i+1}'
                ))
                model.add(tf.keras.layers.Dropout(
                    self.config.dropout_rate,
                    name=f'dropout_{i+1}'
                ))
            if output_shape == 1:
                model.add(tf.keras.layers.Dense(1, activation='linear', name='output'))
                loss = 'mse'
                metrics = ['mae']
            else:
                model.add(tf.keras.layers.Dense(output_shape, activation='softmax', name='output'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss=loss,
                metrics=metrics
            )
            log.info("Model compiled successfully", 
                    total_params=model.count_params(),
                    trainable_params=sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
            return model
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              class_weight: Optional[Dict[int, float]] = None) -> TrainingResult:
        """Train the model with the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weight: Class weights for imbalanced data (optional)
            
        Returns:
            TrainingResult with trained model and metrics
        """
        with self.logger.operation_context("ModelTrainer", "train") as log:
            start_time = time.time()
            input_shape = X_train.shape[1:]
            output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
            model = self.build_model(input_shape, output_shape)
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
                log.info("Using provided validation data", 
                        val_samples=len(X_val))
            else:
                val_split = 0.2
                log.info("Using validation split from training data", 
                        validation_split=val_split)
            if class_weight is None and output_shape > 1:
                class_weight = self._compute_class_weights(y_train)
                log.info("Computed class weights for imbalanced data", 
                        class_weights=class_weight)
            callbacks = self.get_callbacks()
            log.info("Starting model training", 
                    train_samples=len(X_train),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size)
            history = model.fit(
                X_train, y_train,
                validation_data=validation_data,
                validation_split=0.2 if validation_data is None else 0.0,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
            training_time = time.time() - start_time
            final_metrics = {}
            for metric_name, values in history.history.items():
                if values:
                    final_metrics[f"final_{metric_name}"] = float(values[-1])
            log.info("Training completed", 
                    training_time=training_time,
                    final_metrics=final_metrics)
            return TrainingResult(
                model=model,
                history=history.history,
                final_metrics=final_metrics,
                training_time=training_time
            )
    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Get list of training callbacks.
        
        Returns:
            List of configured Keras callbacks
        """
        callbacks = []
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.early_stopping_patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        return callbacks
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        if len(y.shape) > 1:
            y_flat = y.flatten() if y.shape[1] == 1 else y[:, 0]
        else:
            y_flat = y
        classes = np.unique(y_flat)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_flat
        )
        return {int(cls): float(weight) for cls, weight in zip(classes, weights)}
class ModelEvaluator:
    """Evaluates trained models and computes metrics."""
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """Initialize the model evaluator.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger or StructuredLogger(__name__)
    def evaluate(self, 
                 model: tf.keras.Model,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> EvaluationMetrics:
        """Evaluate model performance on test data.
        
        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            EvaluationMetrics with computed scores
        """
        with self.logger.operation_context("ModelEvaluator", "evaluate") as log:
            log.info("Starting model evaluation", test_samples=len(X_test))
            y_pred = model.predict(X_test, verbose=0)
            loss, *other_metrics = model.evaluate(X_test, y_test, verbose=0)
            metrics = EvaluationMetrics(
                loss=float(loss),
                mae=float(other_metrics[0]) if other_metrics else 0.0
            )
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = y_test.flatten() if len(y_test.shape) > 1 else y_test
                classification_metrics = self.compute_classification_metrics(
                    y_true_classes, y_pred_classes
                )
                metrics.precision = classification_metrics['precision']
                metrics.recall = classification_metrics['recall']
                metrics.f1_score = classification_metrics['f1_score']
                metrics.accuracy = classification_metrics.get('accuracy')
                metrics.classification_report = classification_metrics.get('report')
            elif len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                if np.all(np.isin(y_test, [0, 1])):
                    y_pred_binary = (y_pred.flatten() > 0.5).astype(int)
                    y_true_binary = y_test.flatten()
                    classification_metrics = self.compute_classification_metrics(
                        y_true_binary, y_pred_binary
                    )
                    metrics.precision = classification_metrics['precision']
                    metrics.recall = classification_metrics['recall']
                    metrics.f1_score = classification_metrics['f1_score']
                    metrics.accuracy = classification_metrics.get('accuracy')
                    metrics.classification_report = classification_metrics.get('report')
            log.info("Evaluation completed", 
                    loss=metrics.loss,
                    mae=metrics.mae,
                    macro_f1=metrics.get_macro_avg_f1())
            return metrics
    def compute_classification_metrics(self, 
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute detailed classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing precision, recall, f1, and other metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precision_dict = {f"class_{cls}": float(prec) for cls, prec in zip(classes, precision)}
        recall_dict = {f"class_{cls}": float(rec) for cls, rec in zip(classes, recall)}
        f1_dict = {f"class_{cls}": float(f1_val) for cls, f1_val in zip(classes, f1)}
        macro_precision = float(np.mean(precision))
        macro_recall = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))
        precision_dict['macro avg'] = macro_precision
        recall_dict['macro avg'] = macro_recall
        f1_dict['macro avg'] = macro_f1
        weighted_precision = float(np.average(precision, weights=support))
        weighted_recall = float(np.average(recall, weights=support))
        weighted_f1 = float(np.average(f1, weights=support))
        precision_dict['weighted avg'] = weighted_precision
        recall_dict['weighted avg'] = weighted_recall
        f1_dict['weighted avg'] = weighted_f1
        accuracy = float(np.mean(y_true == y_pred))
        report = classification_report(y_true, y_pred, zero_division=0)
        return {
            'precision': precision_dict,
            'recall': recall_dict,
            'f1_score': f1_dict,
            'accuracy': accuracy,
            'report': report
        }


class ModelExporter:
    """Handles model saving and loading operations."""
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """Initialize the model exporter.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger or StructuredLogger(__name__)
    def save_keras(self, 
                   model: tf.keras.Model,
                   path: str,
                   version: str) -> None:
        """Save model in Keras format with versioning.
        
        Args:
            model: Trained Keras model
            path: Base path for saving
            version: Version string for the model
        """
        with self.logger.operation_context("ModelExporter", "save_keras") as log:
            versioned_path = f"{path}_v{version}.h5"
            log.info("Saving model in Keras format", 
                    path=versioned_path,
                    version=version)
            model.save(versioned_path)
            latest_path = f"{path}_latest.h5"
            model.save(latest_path)
            log.info("Model saved successfully", 
                    versioned_path=versioned_path,
                    latest_path=latest_path)
    def save_tfjs(self, model: tf.keras.Model, path: str) -> None:
        """Save model in TensorFlow.js format.
        
        Args:
            model: Trained Keras model
            path: Path for saving TFJS model
        """
        with self.logger.operation_context("ModelExporter", "save_tfjs") as log:
            log.info("Saving model in TensorFlow.js format", path=path)
            try:
                import tensorflowjs as tfjs
                tfjs.converters.save_keras_model(model, path)
                log.info("Model saved in TensorFlow.js format successfully")
            except ImportError:
                log.error("TensorFlow.js not installed. Install with: pip install tensorflowjs")
                raise ImportError("tensorflowjs package required for TFJS export")
    def load_model(self, path: str) -> tf.keras.Model:
        """Load a saved Keras model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        with self.logger.operation_context("ModelExporter", "load_model") as log:
            log.info("Loading model from path", path=path)
            model = tf.keras.models.load_model(path)
            log.info("Model loaded successfully", 
                    total_params=model.count_params())
            return model