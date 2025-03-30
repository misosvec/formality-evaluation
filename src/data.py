import pandas as pd
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None, names=['slovak', 'english'], on_bad_lines='skip', quoting=3) 

def prepare_dataset(formal, informal):
    formal = read_tsv(formal)
    informal = read_tsv(informal)

    formal_shuffled = formal.sample(frac=1, random_state=42).reset_index(drop=True)
    informal_shuffled = informal.sample(frac=1, random_state=42).reset_index(drop=True)

    formal_sample = formal_shuffled.sample(n=2500, random_state=42).reset_index(drop=True)
    informal_sample = informal_shuffled.sample(n=2500, random_state=42).reset_index(drop=True)

    formal_sample['label'] = 1
    informal_sample['label'] = 0

    merged_df = pd.concat([formal_sample, informal_sample], ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged_df

def stem_stop_preprocess(text, language="english"):
    if pd.isna(text):
        return ""
    
    # remove numbers
    text = re.sub(r'\d+', '', text)

    words = text.lower().split()
    
    if language == "slovak":
        stop_words = set(stopwords.words('french'))
        stemmer = SnowballStemmer("french")
    else:
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer("english")
    
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


def bow_df(df):
    # Apply preprocessing to 'slovak' and 'english' columns
    df["slovak"] = df["slovak"].apply(lambda x: stem_stop_preprocess(x, "slovak"))
    df["english"] = df["english"].apply(lambda x: stem_stop_preprocess(x, "english"))

    # Initialize CountVectorizer for both columns
    vectorizer_slovak = CountVectorizer(max_features=1000)
    vectorizer_english = CountVectorizer(max_features=1000)

    # Fit and transform the text data for 'slovak' and 'english' columns
    slovak_bow = vectorizer_slovak.fit_transform(df["slovak"])
    english_bow = vectorizer_english.fit_transform(df["english"])

    # Convert the results to DataFrames
    slovak_bow_df = pd.DataFrame(slovak_bow.toarray(), columns=vectorizer_slovak.get_feature_names_out())
    english_bow_df = pd.DataFrame(english_bow.toarray(), columns=vectorizer_english.get_feature_names_out())

    # Concatenate the BoW DataFrames horizontally (along columns)
    combined_bow_df = pd.concat([slovak_bow_df, english_bow_df], axis=1)

    # Add the label column back to the DataFrame
    combined_bow_df['label'] = df['label']

    # Return the resulting DataFrame with labels
    return combined_bow_df

if __name__ == "__main__":
    formal = 'datasets/sk-en.formal.tsv'
    informal = 'datasets/sk-en.informal.tsv'
    df = prepare_dataset(formal=formal, informal=informal)
    df.to_csv('datasets/sk-en.merged5000.tsv', sep='\t', header=False, index=False)