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
from datetime import datetime
import logging.handlers

# logging configuration
log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# reads and decodes chunks from a file
def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	'''
    Reads and decodes chunks of data from a file reader. This function handles large files by reading in manageable chunks and dealing with UnicodeDecodeErrors by attempting to read more data. If the chunk cannot be decoded within the max_window_size, a UnicodeError is raised.

    Parameters:
    - reader (file reader object): A reader object, typically from a compressed file stream.
    - chunk_size (int): The size of each chunk to read from the file in bytes.
    - max_window_size (int): The maximum number of bytes to attempt reading before giving up on decoding a chunk.
    - previous_chunk (str, optional): Previously read data that couldn't be decoded. Used for recursive calls when decoding fails. Default is None.
    - bytes_read (int, optional): Total number of bytes read so far, used for recursive calls. Default is 0.
    '''
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
	'''
    Generator function that reads and yields lines from a Zstandard compressed file. It uses read_and_decode to handle decompression and decoding of the file's contents. This function is efficient for large files, yielding one line at a time along with the current file position.

    Parameters:
    - file_name (str): Path to the input file, expected to be Zstandard compressed.
    '''
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


if __name__ == "__main__":
	# read CLI arguments
	input_file_path = sys.argv[1]
	output_file_path = sys.argv[2]
	fields = sys.argv[3].split(",")

	# get size of input and initialize counters
	file_size = os.stat(input_file_path).st_size
	file_lines = 0
	file_bytes_processed = 0

	# open output for CSV writing
	line = None
	created = None
	bad_lines = 0
	output_file = open(output_file_path, "w", encoding='utf-8', newline="")
	writer = csv.writer(output_file)
	writer.writerow(fields)

	# read lines from input file
	try:
		for line, file_bytes_processed in read_lines_zst(input_file_path):
			try:
				# parse each line as JSON and extract fields
				obj = json.loads(line)
				output_obj = []
				for field in fields:
					output_obj.append(str(obj[field]).encode("utf-8", errors='replace').decode())
				writer.writerow(output_obj)
				created = datetime.utcfromtimestamp(int(obj['created_utc']))
			except json.JSONDecodeError as err:
				# increment bad lines counter if JSON decoding fails
				bad_lines += 1
			file_lines += 1

			# log process every 100,000 lines
			if file_lines % 100000 == 0:
				log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
	except KeyError as err:
		# log key missing error
		log.info(f"Object has no key: {err}")
		log.info(line)
	except Exception as err:
		# log any other errors
		log.info(err)
		log.info(line)

	# close output file and log completion
	output_file.close()
	log.info(f"Complete : {file_lines:,} : {bad_lines:,}")