# Reddit AITA Dataset Creation
The dataset creation process is as follows:
1. initial dataset dumps were collected using the reddit pushshift API
    - https://www.reddit.com/r/pushshift/comments/1akrhg3/separate_dump_files_for_the_top_40k_subreddits/
    - https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10
    - include S3 links to AmItheAsshole_comments.zst and AmItheAsshole_submissions.zst for ease of use

2. dataset created from dumps
    - filtering
    - creation via merging

3. saved to HF hub with no direct train/test split
    - is this possible?

4. analysis



