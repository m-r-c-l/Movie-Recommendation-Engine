import pandas as pd

# placeholder script if we need preprocessing, for now it just converts
# the blob of input genre into different binary columns per subgenre




# dict to df
def dict_to_df(data: dict):
    return pd.DataFrame(data)


# enhances input dataframe to include separate columns for subgenres
def preprocess_features(data: dict):
    """
    Cleans input column's genre. Identifies all subgenres. Creates new columns
    for each subgenre and returns the preprocessed df
    """
    # dict to df
    raw_df = dict_to_df(data)

    # cleaning genres
    raw_df["genres"] = raw_df["genres"].str.replace(r'[\[\]"]', '', regex=True)

    # finding all subgenres
    all_subgenres = set()
    raw_df['genres'].str.split(',').apply(all_subgenres.update)

    # creating new columns per subgenre
    for subgenre in all_subgenres:
        raw_df[subgenre] = raw_df['genres'].apply(lambda x: 1 if subgenre in x.split(',') else 0) # lambda looks into the list created by x.split()

    return raw_df
