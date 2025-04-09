import zstandard
import os
import json
import sys
import csv
from datetime import datetime
import logging.handlers
from typing import List, Optional, Union
from dataclasses import dataclass

@dataclass
class FilterConfig:
    """Configuration class for filtering parameters"""
    input_file: str
    output_file: str
    output_format: str = "zst"  # Options: zst, txt, csv
    single_field: Optional[str] = None
    min_score: float = float('-inf')
    max_score: float = float('inf')
    write_bad_lines: bool = False
    date_filtering: bool = False
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    score_filtering: bool = False
    field: Optional[str] = None
    values: List[str] = None
    values_file: Optional[str] = None
    exact_match: bool = False

class DataFilter:
    def __init__(self):
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging settings"""
        self.log = logging.getLogger("bot")
        self.log.setLevel(logging.INFO)
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        
        # Stream handler
        log_str_handler = logging.StreamHandler()
        log_str_handler.setFormatter(log_formatter)
        self.log.addHandler(log_str_handler)
        
        # File handler
        if not os.path.exists("logs"):
            os.makedirs("logs")
        log_file_handler = logging.handlers.RotatingFileHandler(
            os.path.join("logs", "bot.log"), 
            maxBytes=1024*1024*16, 
            backupCount=5
        )
        log_file_handler.setFormatter(log_formatter)
        self.log.addHandler(log_file_handler)

    def _write_line_zst(self, handle, line: str):
        """Write a line to a zstd-compressed file"""
        handle.write(line.encode('utf-8'))
        handle.write("\n".encode('utf-8'))

    def _write_line_json(self, handle, obj: dict):
        """Write a JSON object to a file"""
        handle.write(json.dumps(obj))
        handle.write("\n")

    def _write_line_single(self, handle, obj: dict, field: str):
        """Write a single field from a JSON object"""
        if field in obj:
            handle.write(obj[field])
        else:
            self.log.info(f"{field} not in object {obj['id']}")
        handle.write("\n")

    def _write_line_csv(self, writer, obj: dict, is_submission: bool):
        """Write a line to a CSV file"""
        output_list = [
            str(obj['score']),
            datetime.fromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d"),
            obj['title'] if is_submission else None,
            f"u/{obj['author']}",
            f"https://www.reddit.com{obj['permalink']}"
        ]
        
        if is_submission:
            if obj['is_self']:
                output_list.append(obj.get('selftext', ''))
            else:
                output_list.append(obj['url'])
        else:
            output_list.append(obj['body'])
            
        writer.writerow([item for item in output_list if item is not None])

    def _read_and_decode(self, reader, chunk_size: int, max_window_size: int, 
                        previous_chunk=None, bytes_read: int = 0) -> str:
        """Read and decode data from a zstd-compressed stream"""
        chunk = reader.read(chunk_size)
        bytes_read += chunk_size
        
        if previous_chunk is not None:
            chunk = previous_chunk + chunk
            
        try:
            return chunk.decode()
        except UnicodeDecodeError:
            if bytes_read > max_window_size:
                raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
            self.log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
            return self._read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

    def _read_lines_zst(self, file_name: str):
        """Generator to read lines from a zstd-compressed file"""
        with open(file_name, 'rb') as file_handle:
            buffer = ''
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
            
            while True:
                chunk = self._read_and_decode(reader, 2**27, (2**29) * 2)
                if not chunk:
                    break
                    
                lines = (buffer + chunk).split("\n")
                for line in lines[:-1]:
                    yield line.strip(), file_handle.tell()
                buffer = lines[-1]
                
            reader.close()

    def process_file(self, config: FilterConfig):
        """
        Process a file according to the provided configuration
        
        Args:
            config: FilterConfig object containing all filtering parameters
        """
        output_path = f"{config.output_file}.{config.output_format}"
        is_submission = "submission" in config.input_file
        self.log.info(f"Input: {config.input_file} : Output: {output_path} : Is submission {is_submission}")
        
        # Setup output handle and writer
        writer = None
        if config.output_format == "zst":
            handle = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))
        elif config.output_format == "txt":
            handle = open(output_path, 'w', encoding='UTF-8')
        elif config.output_format == "csv":
            handle = open(output_path, 'w', encoding='UTF-8', newline='')
            writer = csv.writer(handle)
        else:
            self.log.error(f"Unsupported output format {config.output_format}")
            sys.exit()

        # Process values
        values = []
        if config.values_file:
            with open(config.values_file, 'r') as values_handle:
                values = [value.strip().lower() for value in values_handle]
            self.log.info(f"Loaded {len(values)} from values file {config.values_file}")
        else:
            values = [value.lower() for value in (config.values or [])]

        file_size = os.stat(config.input_file).st_size
        created = None
        matched_lines = 0
        bad_lines = 0
        total_lines = 0

        for line, file_bytes_processed in self._read_lines_zst(config.input_file):
            total_lines += 1
            if total_lines % 100000 == 0:
                self.log.info(
                    f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : "
                    f"{matched_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:"
                    f"{(file_bytes_processed / file_size) * 100:.0f}%"
                )

            try:
                obj = json.loads(line)
                created = datetime.utcfromtimestamp(int(obj['created_utc']))
                
                # Apply filters
                if config.date_filtering:
                    if created < config.from_date or created > config.to_date:
                        continue

                if config.score_filtering:
                    score = int(obj['score'])
                    if score < config.min_score or score > config.max_score:
                        continue

                if config.field is not None:
                    field_value = obj[config.field].lower()
                    if not any(
                        value == field_value if config.exact_match else value in field_value
                        for value in values
                    ):
                        continue

                # Write output
                matched_lines += 1
                if config.output_format == "zst":
                    self._write_line_zst(handle, line)
                elif config.output_format == "csv":
                    self._write_line_csv(writer, obj, is_submission)
                elif config.output_format == "txt":
                    if config.single_field is not None:
                        self._write_line_single(handle, obj, config.single_field)
                    else:
                        self._write_line_json(handle, obj)

            except (KeyError, json.JSONDecodeError) as err:
                bad_lines += 1
                if config.write_bad_lines:
                    if isinstance(err, KeyError):
                        self.log.warning(f"Key {config.field} is not in the object: {err}")
                    elif isinstance(err, json.JSONDecodeError):
                        self.log.warning(f"Line decoding failed: {err}")
                    self.log.warning(line)

        handle.close()
        self.log.info(f"Complete : {total_lines:,} : {matched_lines:,} : {bad_lines:,}")

def filter_data(config: Union[FilterConfig, dict]):
    """
    Main entry point for filtering data
    
    Args:
        config: Either a FilterConfig object or a dictionary of configuration parameters
    """
    if isinstance(config, dict):
        config = FilterConfig(**config)
        
    data_filter = DataFilter()
    
    # Handle directory input
    input_files = []
    if os.path.isdir(config.input_file):
        if not os.path.exists(config.output_file):
            os.makedirs(config.output_file)
        for file in os.listdir(config.input_file):
            if not os.path.isdir(file) and file.endswith(".zst"):
                input_name = os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0]
                input_files.append((
                    os.path.join(config.input_file, file),
                    os.path.join(config.output_file, input_name)
                ))
    else:
        input_files.append((config.input_file, config.output_file))

    data_filter.log.info(f"Processing {len(input_files)} files")
    for file_in, file_out in input_files:
        file_config = FilterConfig(
            input_file=file_in,
            output_file=file_out,
            **{k: v for k, v in vars(config).items() if k not in ['input_file', 'output_file']}
        )
        data_filter.process_file(file_config)

# Example usage:
if __name__ == "__main__":
    config = FilterConfig(
        input_file="raw-dumps/AmItheAsshole_submissions.zst",
        output_file="filtered_submissions",
        output_format="zst",
        min_score=50,
        score_filtering=True,
        date_filtering=True,
        from_date=datetime.strptime("2019-01-01", "%Y-%m-%d"),
        to_date=datetime.strptime("2023-01-01", "%Y-%m-%d")
    )
    filter_data(config)