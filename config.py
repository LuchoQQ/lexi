import os
import json
import logging
from typing import Any, Dict, Optional
from dotenv import load_dotenv  # Import dotenv to load environment variables
from src.exceptions import ConfigurationError

class Config:
    """Configuration handler for the legal assistant."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config from file or environment variables.
        
        Args:
            config_path: Path to JSON config file (optional)
        """
        self.config = {}
        
        # Load environment variables from .env file
        load_dotenv()  # This will read the .env file in the current directory
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                raise ConfigurationError(f"Error loading config file: {e}")
        
        # Load from environment variables (override file settings)
        self._load_from_env()
        
        # Validate required settings
        self._validate_config()
        
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API Keys
        if os.getenv("OPENAI_API_KEY"):
            self.config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
            
        # Model settings
        if os.getenv("EMBEDDING_MODEL"):
            self.config["embedding_model"] = os.getenv("EMBEDDING_MODEL")
        if os.getenv("RERANK_MODEL"):
            self.config["rerank_model"] = os.getenv("RERANK_MODEL")
        if os.getenv("GENERATION_MODEL"):
            self.config["generation_model"] = os.getenv("GENERATION_MODEL")
            
        # Database settings
        if os.getenv("CHROMA_DB_DIR"):
            self.config["chroma_db_dir"] = os.getenv("CHROMA_DB_DIR")
            
        # Processing settings
        if os.getenv("BATCH_SIZE"):
            try:
                self.config["batch_size"] = int(os.getenv("BATCH_SIZE"))
            except ValueError:
                logging.warning("Invalid BATCH_SIZE, using default value")
    
    def _validate_config(self):
        """Validate that all required configuration is present."""
        required_keys = ["openai_api_key"]
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration: {', '.join(missing_keys)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary syntax.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If the key doesn't exist
        """
        return self.config[key]
