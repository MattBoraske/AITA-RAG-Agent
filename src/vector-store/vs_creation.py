"""
Vector Store Creation Script

This script creates a Pinecone vector store from a HuggingFace dataset.
It serves as the main entry point for vector store creation, delegating
the core functionality to VectorStoreUtility.
"""

import os
import logging
import sys
import pandas as pd
from dataclasses import dataclass
import argparse
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from .vs_util import VectorStoreUtility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vs_creation.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for vector store creation."""
    dataset_name: str
    index_name: str
    embed_provider: str
    embed_endpoint: str
    remove_info_class: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.dataset_name:
            raise ValueError("Dataset name cannot be empty")
        if not self.index_name:
            raise ValueError("Index name cannot be empty")
        if self.embed_provider not in ["openai"]:  # Add more providers as they're supported
            raise ValueError(f"Unsupported embedding provider: {self.embed_provider}")
        if self.embed_provider == "openai" and self.embed_endpoint not in [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]:
            raise ValueError(f"Unsupported OpenAI embedding endpoint: {self.embed_endpoint}")

def parse_arguments() -> VectorStoreConfig:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create a Pinecone vector store from HuggingFace dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='MattBoraske/reddit-AITA-submissions-and-comments-multiclass',
        help='HuggingFace dataset name'
    )
    
    parser.add_argument(
        '--index-name',
        type=str,
        default='aita-text-embedding-3-large',
        help='Name of the Pinecone vector store index'
    )
    
    parser.add_argument(
        '--embed-provider',
        type=str,
        default='openai',
        choices=['openai'],  # Add more providers as they're supported
        help='Embedding model provider'
    )
    
    parser.add_argument(
        '--embed-endpoint',
        type=str,
        default='text-embedding-3-large',
        choices=['text-embedding-3-small', 'text-embedding-3-large'],
        help='Embedding model endpoint'
    )
    
    parser.add_argument(
        '--keep-info',
        action='store_true',
        help='Keep INFO classifications in the dataset'
    )
    
    args = parser.parse_args()
    
    config = VectorStoreConfig(
        dataset_name=args.dataset,
        index_name=args.index_name,
        embed_provider=args.embed_provider,
        embed_endpoint=args.embed_endpoint,
        remove_info_class=not args.keep_info
    )
    
    return config

def load_environment() -> None:
    """Load environment variables and authenticate with services."""
    try:
        # Load environment variables
        env_path = find_dotenv()
        if not env_path:
            raise EnvironmentError("No .env file found")
        load_dotenv(env_path)
        
        # Authenticate with HuggingFace
        huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        if not huggingface_token:
            raise EnvironmentError("HUGGINGFACE_TOKEN not found in environment variables")
        login(token=huggingface_token)
        
        # Verify other required environment variables
        required_vars = ['PINECONE_API_KEY', 'OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
    except Exception as e:
        logger.error(f"Failed to load environment: {str(e)}")
        raise

def create_dataset(config: VectorStoreConfig) -> pd.DataFrame:
    """
    Load and preprocess the dataset from HuggingFace.
    
    Args:
        config: Vector store configuration
        
    Returns:
        Preprocessed pandas DataFrame
    """
    try:
        logger.info(f"Loading dataset: {config.dataset_name}")
        dataset = load_dataset(config.dataset_name)
        df = dataset['train'].to_pandas()
        
        initial_rows = len(df)
        logger.info(f"Initial dataset size: {initial_rows} rows")
        
        # Remove INFO classifications if specified
        if config.remove_info_class:
            df = df[df['top_comment_1_classification'] != 'INFO']
            logger.info(f"Removed {initial_rows - len(df)} rows with INFO classification")
        
        # Replace None values
        vs_util = VectorStoreUtility()
        df = vs_util.replace_none_values(df)
        
        logger.info(f"Final dataset size: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise

def main() -> None:
    """Main execution function."""
    try:
        # Parse arguments and validate configuration
        config = parse_arguments()
        config.validate()
        
        # Load environment variables and authenticate
        load_environment()
        
        # Create VectorStoreUtility instance
        vs_util = VectorStoreUtility()
        
        # Load and preprocess dataset
        dataset_df = create_dataset(config)
        
        # Create vector store
        logger.info(f"Creating Pinecone Vector Store: {config.index_name}")
        
        # Convert dataframe to documents
        documents = vs_util.convert_df_to_documents(dataset_df)
        logger.info(f"Created {len(documents)} documents from dataset")
        
        # Create Pinecone vector store index
        vs_index = vs_util.create_pinecone_vs_index(
            index_name=config.index_name,
            documents=documents,
            embed_model_provider=config.embed_provider,
            embed_model_endpoint=config.embed_endpoint
        )
        
        logger.info(f"Successfully created Pinecone Vector Store: {config.index_name}")
        
    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
         

if __name__ == '__main__':
    main()