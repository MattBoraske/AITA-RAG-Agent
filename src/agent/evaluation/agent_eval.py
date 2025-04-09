import asyncio
import os
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset
import json
from ..agents import AITA_Basic_Agent, AITA_RAG_Agent
from .eval_util import Evaluation_Utility
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from multiprocessing import freeze_support

from phoenix.otel import register


def setup_logging(results_directory, log_level=logging.INFO):
    """Set up logging configuration"""

    # create log file
    log_filename = os.path.join(results_directory, f'agent_eval.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run AITA Agent evaluation with custom parameters')
    
    # Add logging level argument
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    
    # Workflow parameters
    parser.add_argument('--timeout', type=int, default=900,
                      help='Timeout in seconds (default: 900)')
    parser.add_argument('--llm-provider', type=str, default='openai',
                        help='LLM provider (default: openai)')
    parser.add_argument('--llm-endpoint', type=str, default='gpt-4o-mini-2024-07-18',
                        help='LLM endpoint')
    parser.add_argument('--embedding-provider', type=str, default='openai',
                        help='Embedding model provider')
    parser.add_argument('--embedding-endpoint', type=str, default='text-embedding-3-large',
                      help='Embedding model endpoint')
    parser.add_argument('--pinecone-index', type=str, default='aita-text-embedding-3-large-v2',
                      help='Pinecone vector index name')
    parser.add_argument('--docs-to-retrieve', type=int, default=5,
                      help='Number of documents to retrieve')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, 
                      default='MattBoraske/reddit-AITA-submissions-and-comments-multiclass',
                      help='HuggingFace dataset path')
    parser.add_argument('--sampling', type=str, default='complete',
                        choices=['complete', 'balanced', 'weighted'],
                        help='Dataset sampling strategy')
    parser.add_argument('--complete-total-samples-start', type=int, default=None,
                        help='Total number of samples to use for complete sampling (start)')
    parser.add_argument('--complete-total-samples-end', type=int, default=None,
                        help='Total number of samples to use for complete sampling (end)')
    parser.add_argument('--balanced-samples-per-class', type=int, default=1,
                      help='Number of samples per class for balanced sampling')
    parser.add_argument('--weighted-total-samples', type=int, default=1,
                        help='Total number of samples to use for weighted sampling')
    
    # Evaluation parameters
    parser.add_argument('--run-eval', action='store_true',
                        help='Run evaluation of responses')
    parser.add_argument('--eval-type', type=str, default='BASIC',
                        choices=['BASIC', 'RAG'],
                        help='Type of evaluation to perform')
    parser.add_argument('--phoenix-project', type=str, default='default',
                        help='Phoenix project name to log LLM traces to')

    # Output file parameters
    parser.add_argument('--responses-file', type=str, default='responses.json')
    parser.add_argument('--classification-report-filepath', type=str, 
                      default='classification_report.txt')
    parser.add_argument('--confusion-matrix-filepath', type=str, 
                      default='confusion_matrix.png')
    parser.add_argument('--mcc-filepath', type=str, default='mcc_score.json')
    parser.add_argument('--rouge-filepath', type=str, default='rouge_score.json')
    parser.add_argument('--bleu-filepath', type=str, default='bleu_score.json')
    parser.add_argument('--comet-filepath', type=str, default='comet_score.json')
    parser.add_argument('--toxicity-stats-filepath', type=str, 
                      default='toxicity_stats.json')
    parser.add_argument('--toxicity-plot-filepath', type=str, 
                      default='toxicity_plot.png')
    parser.add_argument('--retrieval-eval-filepath', type=str, 
                      default='retrieval_eval.json')
    parser.add_argument('--retrieval-eval-summary-filepath', type=str, 
                      default='retrieval_eval_summary.json')
    
    return parser.parse_args()

def run_evaluation(args, logger, results_directory):
    """Run the evaluation process with logging"""
    logger.info("Starting evaluation process")
    logger.debug(f"Evaluation parameters: {vars(args)}")
    
    try:
        # Initialize evaluation utility
        logger.info("Initializing Evaluation Utility")
        eval_util = Evaluation_Utility()

        # Initialize workflow
        if args.eval_type == 'BASIC':
            logger.info("Initializing AITA Basic Agent workflow")
            workflow = AITA_Basic_Agent(
                timeout=args.timeout,
                llm_provider=args.llm_provider,
                llm_endpoint=args.llm_endpoint
            )
        elif args.eval_type == 'RAG':
            logger.info("Initializing AITA RAG Agent workflow")
            workflow = AITA_RAG_Agent(
                timeout=args.timeout,
                llm_provider=args.llm_provider,
                llm_endpoint=args.llm_endpoint,
                embedding_provider=args.embedding_provider,
                embedding_model_endpoint=args.embedding_endpoint,
                pinecone_vector_index=args.pinecone_index,
                docs_to_retrieve=args.docs_to_retrieve,
            )
        else:
            raise ValueError(f"Invalid evaluation type: {args.eval_type}")
        
        # Load and prepare dataset
        logger.info(f"Loading dataset from {args.dataset}")
        hf_dataset = load_dataset(args.dataset)
        AITA_test_df = hf_dataset['test'].to_pandas()
        
        logger.debug("Filtering out INFO classifications")
        AITA_test_df = AITA_test_df[AITA_test_df['top_comment_1_classification'] != 'INFO']
        logger.info(f"Dataset size after filtering: {len(AITA_test_df)}")
        
        # Create test set
        logger.info("Creating test set")
        test_set = eval_util.create_test_set(
            df=AITA_test_df,
            sampling=args.sampling,
            total_samples_start = args.complete_total_samples_start,
            total_samples_end = args.complete_total_samples_end,
            balanced_samples_per_class=args.balanced_samples_per_class,
            weighted_total_samples=args.weighted_total_samples
        )
        
        logger.info(f"Final test set size: {len(test_set)}")
        
        # Collect responses
        logger.info("Starting response collection")
        responses = asyncio.run(eval_util.collect_responses(workflow, test_set, args.eval_type))
        logger.info(f"Collected {len(responses)} responses")

        # Save responses
        responses_path = os.path.join(results_directory, args.responses_file)
        logger.info(f"Saving responses to {responses_path}")
        with open(responses_path, 'w') as f:
            json.dump(responses, f)

        # Run evaluation
        if args.run_eval:
            logger.info("Starting evaluation of responses")
            eval_util.evaluate(
                responses=responses,
                results_directory=results_directory,
                eval_type = args.eval_type,
                classification_report_filepath=args.classification_report_filepath,
                confusion_matrix_filepath=args.confusion_matrix_filepath,
                mcc_filepath=args.mcc_filepath,
                rouge_filepath=args.rouge_filepath,
                bleu_filepath=args.bleu_filepath,
                comet_filepath=args.comet_filepath,
                toxicity_stats_filepath=args.toxicity_stats_filepath,
                toxicity_plot_filepath=args.toxicity_plot_filepath,
                retrieval_eval_filepath=args.retrieval_eval_filepath,
                retrieval_eval_summary_filepath=args.retrieval_eval_summary_filepath
            )
        else:
            logger.info("Skipping evaluation of responses")
            
        logger.info("Evaluation process completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation process: {str(e)}", exc_info=True)
        raise

def setup_phoenix_telemetry(logger, phoenix_project):
    """Set up Phoenix LLM tracing telemetry"""
    logger.info("Setting up telemetry")
    try:
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
        logger.debug("Added Phoenix API key to environment")
        
        tracer_provider = register(
            project_name=phoenix_project, # Default is 'default'
            endpoint="https://app.phoenix.arize.com/v1/traces",
        )

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Telemetry setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up Phoenix telemetry: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set up results directory
    results_directory = os.path.join("eval_results", datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    os.makedirs(results_directory, exist_ok=True)
    
    # log args to result directory
    with open(os.path.join(results_directory, 'eval_args.json'), 'w') as f:
        json.dump(vars(args), f)

    # Setup logging with specified level
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(results_directory, log_level)
    
    logger.info("Starting AITA Agent evaluation script")
    
    try:
        load_dotenv(find_dotenv())
        freeze_support()
        setup_phoenix_telemetry(logger, args.phoenix_project)
        run_evaluation(args, logger, results_directory)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Script execution completed")