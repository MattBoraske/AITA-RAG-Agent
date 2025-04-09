"""
Vector Store Utility Module

This module provides the VectorStoreUtility class for managing vector store operations,
including data preprocessing, document conversion, and vector store creation.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pprint import pformat

import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    dimensions: int
    supported_providers: List[str]
    supported_endpoints: List[str]

    @classmethod
    def get_config(cls, provider: str, endpoint: str) -> 'EmbeddingConfig':
        """
        Get embedding configuration for specific provider and endpoint.
        
        Args:
            provider: Embedding provider name
            endpoint: Specific model endpoint
            
        Returns:
            EmbeddingConfig for the specified provider and endpoint
        """
        if provider == "openai":
            return cls(
                dimensions=3072 if endpoint == "text-embedding-3-large" else 1536,
                supported_providers=["openai"],
                supported_endpoints=["text-embedding-3-small", "text-embedding-3-large"]
            )
        raise ValueError(f"Unsupported embedding provider: {provider}")

class VectorStoreUtility:
    """Utility class for vector store operations."""
    
    def __init__(self):
        """Initialize VectorStoreUtility."""
        self._pinecone_client: Optional[Pinecone] = None
        
    @property
    def pinecone_client(self) -> Pinecone:
        """
        Lazy initialization of Pinecone client.
        
        Returns:
            Initialized Pinecone client
        """
        if self._pinecone_client is None:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise EnvironmentError("PINECONE_API_KEY not found in environment variables")
            self._pinecone_client = Pinecone(api_key=api_key)
        return self._pinecone_client

    def replace_none_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace None values in the DataFrame with appropriate defaults.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with None values replaced
        """
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Replace None in top comment columns
            comment_cols = [col for col in df.columns if col.startswith('top_comment_') 
                          and not col.endswith('classification')]
            for col in comment_cols:
                df[col] = df[col].fillna('No Comment')
                logger.debug(f"Replaced None values in column: {col}")
            
            # Replace None in classification columns
            classification_cols = [col for col in df.columns if col.endswith('classification')]
            for col in classification_cols:
                df[col] = df[col].fillna('No Classification')
                logger.debug(f"Replaced None values in column: {col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error replacing None values: {str(e)}")
            raise

    def convert_df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame rows to LlamaIndex Documents.
        
        Args:
            df: DataFrame with submission and comment data
            
        Returns:
            List of LlamaIndex Document objects
        """
        try:
            documents = []
            total_rows = len(df)
            
            for idx, row in df.iterrows():
                if idx > 0 and idx % 1000 == 0:
                    logger.info(f"Processed {idx}/{total_rows} rows")
                
                # Construct document text
                text = self._create_document_text(row)
                
                # Create metadata
                metadata = self._create_document_metadata(row)
                
                # Create document
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Successfully converted {len(documents)} rows to documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error converting DataFrame to documents: {str(e)}")
            raise

    def _create_document_text(self, row: pd.Series) -> str:
        """
        Create document text from row data.
        
        Args:
            row: DataFrame row
            
        Returns:
            Formatted document text
        """
        return (
            f"{row['submission_title']}\n\n"
            f"{row['submission_text']}\n\n"
            #f"Correct Classification: {row['top_comment_1_classification']}\n\n"
            #f"Correct Justification: {row['top_comment_1']}"
        )

    def _create_document_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Create document metadata from row data.
        
        Args:
            row: DataFrame row
            
        Returns:
            Document metadata dictionary
        """
        return {
            'Submission URL': row['submission_url'],
            'Correct Classification': row['top_comment_1_classification'],
            'Correct Justification': row['top_comment_1'],
            # Add additional metadata fields as needed
        }

    def create_pinecone_vs_index(
        self,
        index_name: str,
        documents: List[Document],
        embed_model_provider: str = "openai",
        embed_model_endpoint: str = "text-embedding-3-small"
    ) -> VectorStoreIndex:
        """
        Create a Pinecone vector store index.
        
        Args:
            index_name: Name for the Pinecone index
            documents: List of documents to index
            embed_model_provider: Embedding model provider
            embed_model_endpoint: Specific embedding model endpoint
            
        Returns:
            Created VectorStoreIndex
        """
        try:
            # Get embedding configuration
            embed_config = EmbeddingConfig.get_config(embed_model_provider, embed_model_endpoint)
            
            logger.info(f"Creating Pinecone index: {index_name}")
            logger.info(f"Using embedding model: {embed_model_provider}/{embed_model_endpoint}")
            
            # Create Pinecone index
            self.pinecone_client.create_index(
                name=index_name,
                dimension=embed_config.dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Get index instance
            pinecone_index = self.pinecone_client.Index(index_name)
            
            # Create vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Initialize embedding model
            embed_model = self._initialize_embedding_model(embed_model_provider, embed_model_endpoint)
            
            # Configure node parser
            parser = SentenceSplitter(chunk_size=8192)
            
            # Create and populate index
            logger.info("Creating vector store index and embedding documents...")
            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=embed_model,
                node_parser=parser,
                show_progress=True
            )
            
            # Log index description
            self._log_index_description(index_name)
            
            return index
            
        except Exception as e:
            logger.error(f"Error creating Pinecone vector store index: {str(e)}")
            raise

    def _initialize_embedding_model(
        self,
        provider: str,
        endpoint: str
    ) -> Any:
        """
        Initialize the embedding model.
        
        Args:
            provider: Embedding model provider
            endpoint: Specific model endpoint
            
        Returns:
            Initialized embedding model
        """
        if provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
            return OpenAIEmbedding(model=endpoint, api_key=api_key)
        
        raise ValueError(f"Unsupported embedding provider: {provider}")

    def _log_index_description(self, index_name: str) -> None:
        """
        Log Pinecone index description.
        
        Args:
            index_name: Name of the Pinecone index
        """
        try:
            index_description = self.pinecone_client.describe_index(index_name)
            logger.info("Vector Store Index Description:")
            logger.info(pformat(index_description))
        except Exception as e:
            logger.warning(f"Failed to get index description: {str(e)}")