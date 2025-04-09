# main.py
import dc_script
import df_script

# run the filtering script to get the raw datafiles needed
 # 1. filtered submissions on score threshold (currently 50)
 # 2. submission ids for these filtered submissions
 # 3. filtered comments on score threshold (currently 10)
 # 4. get "top-level" comments for each filtered submission from filtered comments

 # at the end of this we have two important files:
    # 1. filtered submissions (zst or csv?)
    # 2. top level filtered comments the filtered submissions

# now run the dataset creation script to create the final dataset
    # saves dataset locally as csv
      # ultimately save on S3 alongside raw dumps
         # storing intermediate ones isn't necessary and messy

# LOG EVERYTHING 

# create a script to push dataset to HF hub
    # seperates train/test split concern

# should include dataset analysis in another script

def create_AITA_dataset():
    print("Running raw datafile filtering script...")
    df_script.main()  # Assuming each script has a main() function
    
    print("Creating dataset from refined datafile...")
    dc_script.main()

if __name__ == "__main__":
    create_AITA_dataset()