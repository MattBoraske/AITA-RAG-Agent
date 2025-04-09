#!/usr/bin/env python3
"""
Reddit AITA Dataset Creation Script

This script processes Reddit AITA (Am I The A-hole) submissions and their comments
to create a dataset for analysis. It processes submissions with at least 50 score
and their top comments with at least 10 score.

Input files required:
1. submissions_2019_to_2022_at_least_50_score.csv: AITA submissions with ≥50 score
2. top_level_comments_2019_to_2022_at_least_10_comment_score_at_least_50_submission_score.csv: 
   Top level comments with ≥10 score for submissions with ≥50 score

Output files:
1. CSV and ZST files containing AITA submissions with their top 10 comments
"""

import pandas as pd
import zstandard as zstd
from pathlib import Path

def load_and_process_submissions(file_path):
    """Load and process the submissions dataframe."""
    submissions_df = pd.read_csv(file_path)
    
    # Filter for relevant AITA decision classes
    valid_flairs = [
        'Asshole',
        'Not the A-hole',
        'No A-holes here',
        'Everyone Sucks',
        'Not enough info'
    ]
    submissions_df = submissions_df[submissions_df['link_flair_text'].isin(valid_flairs)]
    
    # Rename columns for clarity
    submissions_df = submissions_df.rename(columns={
        'id': 'submission_id',
        'link_flair_text': 'decision',
        'score': 'submission_score',
        'title': 'submission_title',
        'selftext': 'submission_text',
        'url': 'submission_url'
    })
    
    return submissions_df

def load_and_process_comments(file_path):
    """Load and process the comments dataframe."""
    comments_df = pd.read_csv(file_path)
    
    # Strip 't3_' prefix from link_id
    comments_df['link_id'] = comments_df['link_id'].str.slice(3)
    
    # Rename columns for clarity
    comments_df = comments_df.rename(columns={
        'id': 'comment_id',
        'score': 'comment_score',
        'body': 'comment_text'
    })
    
    return comments_df

def merge_submissions_and_comments(submissions_df, comments_df):
    """Merge submissions with their top 10 comments."""
    # Merge submissions with comments
    merged_df = submissions_df.merge(comments_df, left_on='submission_id', right_on='link_id')
    merged_df = merged_df.drop('link_id', axis=1)
    
    # Get top 10 comments for each submission
    top_10_comments = merged_df.groupby('submission_id').apply(
        lambda x: x.nlargest(10, 'comment_score')['comment_text'].tolist()
    )
    
    # Convert to dataframe with numbered comment columns
    top_10_comments_df = pd.DataFrame(
        top_10_comments.tolist(), 
        index=top_10_comments.index
    ).add_prefix('comment_')
    
    # Merge submissions with their top comments
    return submissions_df.merge(top_10_comments_df, on='submission_id')

def clean_and_format_dataset(df):
    """Clean and format the final dataset."""
    # Filter out deleted/removed/null content
    df = df[
        (df['submission_text'] != '[deleted]') & 
        (df['comment_0'] != '[deleted]') &
        (df['submission_text'] != '[removed]') & 
        (df['comment_0'] != '[removed]') &
        (df['submission_text'].notnull()) & 
        (df['comment_0'].notnull())
    ]
    
    # Convert timestamps
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    
    # Rename columns
    comment_renames = {
        f'comment_{i}': f'top_comment_{i+1}'
        for i in range(10)
    }
    df = df.rename(columns={
        'created_utc': 'submission_date',
        **comment_renames
    })
    
    # Remove submission_id
    df = df.drop('submission_id', axis=1)
    
    # Reorder columns
    df[['decision', 'submission_title']] = df[['submission_title', 'decision']]
    df = df.rename(columns={
        'decision': 'submission_title',
        'submission_title': 'decision'
    })
    
    df[['submission_score', 'submission_text']] = df[['submission_text', 'submission_score']]
    df = df.rename(columns={
        'submission_score': 'submission_text',
        'submission_text': 'submission_score'
    })
    
    return df

def save_dataset(df, base_filename):
    """Save the dataset in both CSV and ZST formats."""
    # Save as CSV
    csv_path = f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    
    # Compress to ZST
    zst_path = f"{base_filename}.zst"
    with open(csv_path, 'rb') as f_in, open(zst_path, 'wb') as f_out:
        cctx = zstd.ZstdCompressor()
        cctx.copy_stream(f_in, f_out)

def main():
    # Input files
    submissions_file = "submissions_2019_to_2022_at_least_50_score.csv"
    comments_file = "top_level_comments_2019_to_2022_at_least_10_comment_score_at_least_50_submission_score.csv"
    
    # Output base filename
    output_base = "2019_to_2022_submissions_at_least_50_score_top_10_comments"
    
    # Process data
    print("Loading and processing submissions...")
    submissions_df = load_and_process_submissions(submissions_file)
    
    print("Loading and processing comments...")
    comments_df = load_and_process_comments(comments_file)
    
    print("Merging submissions with top comments...")
    merged_df = merge_submissions_and_comments(submissions_df, comments_df)
    
    print("Cleaning and formatting dataset...")
    final_df = clean_and_format_dataset(merged_df)
    
    print("Saving dataset...")
    save_dataset(final_df, output_base)
    
    print("Dataset creation complete!")

if __name__ == "__main__":
    main()