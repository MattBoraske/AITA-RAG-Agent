"""
# script that converts a .zst into a .csv

example usages:
submissions:
    python script.py -i submissions.zst -o comments.csv -f id,link_flair_text,score,title,selftext,url,created_utc
comments:
    python script.py -i comments.zst -o comments.csv -f id,link_id,score,body
submissions+comments (final dataset):

"""
# script that converts a .zst into a .csv
# arguments: inputfile, outputfile, fields

# example usages:
## submissions:
# python zst_to_csv.py submissions_2019_to_2022_above_50_score_take_2.zst submissions_2019_to_2022_above_50_score_take_2.csv id,link_flair_text,score,title,selftext,url,created_utc
## comments:
# python zst_to_csv.py top_level_comments_2022_score_50.zst top_level_comments_2022_score_50.csv id,link_id,score,body

import zstandard
import os
import json
import sys
import csv
import argparse
from datetime import datetime
import logging.handlers

def setup_logging():
    """Configure and return logger"""
    log = logging.getLogger("bot")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())
    return log

def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert ZST file to CSV with specified fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  For submissions:
    %(prog)s -i submissions.zst -o submissions.csv -f id,link_flair_text,score,title,selftext,url,created_utc
  For comments:
    %(prog)s -i comments.zst -o comments.csv -f id,link_id,score,body
        '''
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the input ZST file containing JSON records. Each line should be a valid JSON object.'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path where the output CSV file will be created. Will overwrite if file already exists.'
    )
    
    parser.add_argument(
        '-f', '--fields',
        required=True,
        help='Comma-separated list of fields to extract from each JSON record (e.g., "id,score,title"). These must match the keys in your JSON data.'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        parser.error(f"Input file does not exist: {args.input}")
    
    # Convert fields string to list
    args.fields = args.fields.split(',')
    
    return args

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    """Read and decode chunks from a file"""
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
    """Generator to read lines from a ZST file"""
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]
        reader.close()

def process_file(input_path, output_path, fields, log):
    """Process the ZST file and write to CSV"""
    file_size = os.stat(input_path).st_size
    file_lines = 0
    bad_lines = 0
    created = None
    
    with open(output_path, "w", encoding='utf-8', newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(fields)
        
        try:
            for line, file_bytes_processed in read_lines_zst(input_path):
                try:
                    obj = json.loads(line)
                    output_obj = [
                        str(obj[field]).encode("utf-8", errors='replace').decode()
                        for field in fields
                    ]
                    writer.writerow(output_obj)
                    created = datetime.utcfromtimestamp(int(obj['created_utc']))
                except json.JSONDecodeError:
                    bad_lines += 1
                
                file_lines += 1
                
                if file_lines % 100000 == 0:
                    log.info(
                        f"{created.strftime('%Y-%m-%d %H:%M:%S')} : "
                        f"{file_lines:,} : {bad_lines:,} : "
                        f"{(file_bytes_processed / file_size) * 100:.0f}%"
                    )
                    
        except KeyError as err:
            log.error(f"Object has no key: {err}")
            log.error(f"Problematic line: {line}")
        except Exception as err:
            log.error(f"Unexpected error: {err}")
            log.error(f"Problematic line: {line}")
    
    log.info(f"Complete : {file_lines:,} : {bad_lines:,}")
    return file_lines, bad_lines

def main():
    """Main entry point of the script"""
    log = setup_logging()
    args = parse_arguments()
    
    try:
        total_lines, bad_lines = process_file(
            args.input,
            args.output,
            args.fields,
            log
        )
        sys.exit(0)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()