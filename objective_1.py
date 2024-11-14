import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import emoji
import re
import os
import nltk
import time
import contractions
import gensim.downloader as api

# limit to the lenght of token that should be displayed on the daatsets
pd.options.display.max_colwidth = 500 

#  to extract the stored kaggle api token
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import kaggle
import kagglehub
# # Set the Kaggle config directory
# os.environ['KAGGLE_USERNAME'] = 'mikelkn'
# os.environ['KAGGLE_KEY'] = "7c338d489b83dbbbccfe3de19f1028d6"
# Authenticate with Kaggle API
kaggle.api.authenticate()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from num2words import num2words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

#for sentiment analysis ad cosine similarity
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification #for sentiment analysis
from transformers import  DistilBertTokenizer, DistilBertModel#for calculating embedding for cosine similarity
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.nn.functional import softmax
import torch.nn.functional as F

#for lemmatizing our corpus
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger')
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

#for removing stop words from our corpus
nltk.download('punkt')
nltk.download('stopwords')
stopwords_list = set(stopwords.words("english"))

class Read:
    @staticmethod
    def read_and_filter_dataset (filepath, text_column, max_length=1000):
        data = pd.read_csv(filepath, low_memory=False)
        # Filter rows based on text length
        data = data[data[text_column].str.len() <= max_length]
        return data

class BERTanalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # """For getting the bert embeddings for the cosine similarity calculation"""
        self.tokenizer_emb = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model_emb =  DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model_emb.to(self.device)
        # """For classifying the sentiment in the datasets"""
        self.tokenizer_sent = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model_sent = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
        self.model_sent.to(self.device)
        self.model_sent.eval()

    def preprocess(self, text):
        """Preprocess text for sentiment analysis."""
        return self.tokenizer_sent(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    def get_bert_embeddings(self, texts):
        """Generate BERT embeddings for a list of texts."""
        inputs = self.tokenizer_emb(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_emb(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu()  # Pooling to get sentence-level embeddings, return to CPU for similarity

    def predict(self, text):
        """Predict sentiment."""
        inputs = self.preprocess(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model_sent(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        # Get the class with the highest probability
        max_prob_idx = torch.argmax(probabilities, dim=1).item()
        if max_prob_idx == 1:
            return "Positive"
        else:
            return "Negative"

    def plot_sentiment_distribution(self, sentiment_labels1, sentiment_labels2, name1, name2):
        """Plots sentiment distribution."""
        sentiment_counts_1 = sentiment_labels1.value_counts()# Count the sentiment labels for each dataset
        sentiment_counts_2 = sentiment_labels2.value_counts()
        
        # Create a DataFrame to hold the counts
        sentiment_df = pd.DataFrame({
            'Positive': [sentiment_counts_1.get('Positive', 0), sentiment_counts_2.get('Positive', 0)],
            'Negative': [sentiment_counts_1.get('Negative', 0), sentiment_counts_2.get('Negative', 0)]
        })
        # Set the index to be the dataset labels for easy plotting
        sentiment_df.index = [name1, name2]
        # Plot the bar chart
        sentiment_df.plot(kind='bar', stacked=True, figsize=(5, 3), color=['green', 'red'])
        plt.title('Sentiment Labels Distribution Across Datasets')
        plt.xlabel('Dataset')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title="Sentiment", loc='upper left')
        plt.show()


    def cosine_similarity(self, dataset1, dataset2):
        # Convert DataFrame columns to lists if necessary
        if isinstance(dataset1, pd.DataFrame):
            dataset1 = dataset1.iloc[:, 0].tolist()  # since text is in the first column
        elif isinstance(dataset1, pd.Series):
            dataset1 = dataset1.tolist()
            
        if isinstance(dataset2, pd.DataFrame):
            dataset2 = dataset2.iloc[:, 0].tolist()
        elif isinstance(dataset2, pd.Series):
            dataset2 = dataset2.tolist()
            
        # Generate embeddings for each dataset
        dataset1_embeddings = self.get_bert_embeddings(dataset1)
        dataset2_embeddings = self.get_bert_embeddings(dataset2)
        
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(dataset1_embeddings, dataset2_embeddings)
        
        # Optionally, calculate an average similarity score for an overall metric
        average_similarity = similarity_matrix.mean()
        
        return average_similarity

class XLNetSentimentAnalyzer:
    def __init__(self, model_name='xlnet-base-cased', num_labels=2):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def classify_sentiment(self, text):
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Perform forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits and apply softmax to get probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

        # Get predicted label
        predicted_label = torch.argmax(probabilities, dim=1).item()
        sentiment = "Positive" if predicted_label == 1 else "Negative"
        
        return {
            "text": text,
            "label": predicted_label,
            "sentiment": sentiment,
            "probabilities": probabilities.tolist()
        }

class Preprocessor:
    def __init__(self):

        self.patterns= {
            "multi_space": r' +',
            "user_mention": r'@\w+',
            "new_line": r'\n+',
            "hyperlink": r'https?://\S+|www\.\S+',
            "accented": r'\^[a-zA-Z0-9]+',
            "date": r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s(?:Jan|Feb|Mar|...|Dec)[a-z]*\s?\d{4}?)\b',
            "time": r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[aApP][mM])?\b',
            "special_characters": r'[_"\-%;()|+&=*%.,!?:#$@[\]/]',
            "numbers": r'\d+',
            "amper_begin_pattern" : r'^',
            "punctuations" : r'[^\w\s]'
        }

    def expand_abbrev_profanity(self, profanity_df, text):
        """Replaces misspelled profanities with correct versions based on a DataFrame of abbreviations and their expanded forms."""
        for _, row in profanity_df.iterrows():
            abbr = row['abbrv']  
            long = row['long']   
            text = re.sub(rf'\b{re.escape(abbr)}\b', long, text, flags=re.IGNORECASE)
        return text

        
    def convert_emojis(self, text):
        """Converts emojis to descriptive words."""
        return emoji.demojize(text).replace(":", "")

    def convert_emoticons(self, text):
        """Converts emoticons to descriptive words."""

        emot_dict = { ":‑\)":"Happy face", 
        ":\)":"Happy face", 
        ":-\]":"Happy face",
        ":\]":"Happy face",
        ":-3":"Happy face",
        ":3":"Happy face",
        ":->":"Happy face ",
        ":>":"Happy face",
        "8-\)":"Happy face smiley",
        ":o\)":"Happy face smiley",
        ":-\}":"Happy face smiley",
        ":\}":"Happy face smiley",
        ":-\)":"Happy face smiley",
        ":c\)":"Happy face smiley",
        ":\^\)":"Happy face smiley",
        "=\]":"Happy face smiley",
        "=\)":"Happy face smiley",
        ":‑D":"Laughing",
        ":D":"Laughing",
        "8‑D":"Laughing",
        "8D":"Laughing",
        "X‑D":"Laughing",
        "XD":"Laughing",
        "=D":"Laughing",
        "=3":"Laughing",
        "B\^D":"Laughing",
        ":-\)\)":"Very happy",
        ":‑\(":"pouting",
        ":-\(":"ad",
        ":\(":"sad",
        ":‑c":"Frown",
        ":c":"Frown",
        ":‑<":"pouting",
        ":<":" pouting",
        ":‑\[":"pouting",
        ":\[":"Frown",
        ":-\|\|":"Frown",
        ">:\[":"Frown",
        ":\{":"Frown",
        ":@":"Frown",
        ">:\(":"Frown",
        ":'‑\(":"Crying",
        ":'\(":"Crying",
        ":'‑\)":"Tears of happiness",
        ":'\)":"Tears of happiness",
        "D‑':":"Horror",
        "D:<":"Disgust",
        "D:":"Sadness",
        "D8":"Great dismay",
        "D;":"Great dismay",
        "D=":"Great dismay",
        "DX":"Great dismay",
        ":‑O":"Surprise",
        ":O":"Surprise",
        ":‑o":"Surprise",
        ":o":"Surprise",
        ":-0":"Shock",
        "8‑0":"Yawn",
        ">:O":"Yawn",
        ":-\*":"Kiss",
        ":\*":"Kiss",
        ":X":"Kiss",
        ";‑\)":"smirk",
        ";\)":"smirk",
        "\*-\)":"smirk",
        "\*\)":"smirk",
        ";‑\]":"smirk",
        ";\]":"Wink ",
        ";\^\)":"smirk",
        ":‑,":"Wink",
        ";D":"Wink",
        ":‑P":"Tongue sticking out",
        ":P":"Tongue sticking out",
        "X‑P":"Tongue sticking out",
        "XP":"Tongue sticking out",
        ":‑Þ":"Tongue sticking out",
        ":Þ":"Tongue sticking out",
        ":b":"Tongue sticking out",
        "d:":"Tongue sticking out",
        "=p":"Tongue sticking out",
        ">:P":"Tongue sticking out",
        ":‑/":"annoyed",
        ":/":"annoyed",
        ":-[.]":"Skeptical",
        ">:[(\\\)]":"Skeptical",
        ">:/":"Skeptical",
        ":[(\\\)]":"Skeptical",
        "=/":"annoyed",
        "=[(\\\)]":"Skeptical",
        ":L":"Skeptical",
        "=L":"Skeptical",
        ":S":"annoyed",
        ":‑\|":"Straight face",
        ":\|":"Straight face",
        ":$":"Embarrassed or blushing",
        ":‑x":"Sealed lips",
        ":x":"Sealed lips",
        ":‑#":"Sealed lips",
        ":#":"Sealed lips",
        ":‑&":"Sealed lips",
        ":&":"Sealed lips",
        "O:‑\)":"innocent",
        "O:\)":"Angel",
        "0:‑3":"saint",
        "0:3":"Angel",
        "0:‑\)":"innocent",
        "0:\)":"innocent",
        ":‑b":"playful",
        "0;\^\)":"Angel, saint or innocent",
        ">:‑\)":"devilish",
        ">:\)":"Evil",
        "\}:‑\)":"devilish",
        "\}:\)":"Evil",
        "3:‑\)":"devilish",
        "3:\)":"Evil",
        ">;\)":"devilish",
        "\|;‑\)":"Cool",
        "\|‑O":"Bored",
        ":‑J":"Tongue-in-cheek",
        "#‑\)":"Party all night",
        "%‑\)":"Drunk or confused",
        "%\)":"Drunk or confused",
        ":-###..":"Being sick",
        ":###..":"Being sick",
        "<:‑\|":"Dump",
        "\(>_<\)":"Troubled",
        "\(>_<\)>":"Troubled",
        "\(';'\)":"Baby",
        "\(\^\^>``":"Embarrassed",
        "\(\^_\^;\)":"Sweat drop",
        "\(-_-;\)":"Shy and Sweat drop",
        "\(~_~;\) \(・\.・;\)":"Nervous and Sweat drop",
        "\(-_-\)zzz":"Sleeping",
        "\(\^_-\)":"Wink",
        "\(\(\+_\+\)\)":"Confused",
        "\(\+o\+\)":"Confused",
        "\(o\|o\)":"Ultraman",
        "\^_\^":"Joyful",
        "\(\^_\^\)/":"Joyful",
        "\(\^O\^\)／":"Joyful",
        "\(\^o\^\)／":"Joyful",
        "\(__\)":"Kowtow as a sign of respect",
        "_\(\._\.\)_":"Kowtow as a sign of respect",
        "<\(_ _\)>":"Kowtow as a sign of respect",
        "<m\(__\)m>":"Kowtow as a sign of respect",
        "m\(__\)m":"Kowtow as a sign of respect",
        "m\(_ _\)m":"Kowtow as a sign of respect",
        "\('_'\)":"Sad or Crying",
        "\(/_;\)":"Crying",
        "\(T_T\) \(;_;\)":" Crying",
        "\(;_;":"Crying",
        "\(;_:\)":"Crying",
        "\(;O;\)":"Crying",
        "\(:_;\)":"Crying",
        "\(ToT\)":" Crying",
        ";_;":"Crying",
        ";-;":"Crying",
        ";n;":"Crying",
        ";;":"SCrying",
        "Q\.Q":"Crying",
        "T\.T":"Crying",
        "QQ":"Crying",
        "Q_Q":"Crying",
        "\(-\.-\)":"Shame",
        "\(-_-\)":"Shame",
        "\(一一\)":"Shame",
        "\(；一_一\)":"Shame",
        "\(=_=\)":"Tired",
        "\(=\^\·\^=\)":"cat",
        "\(=\^\·\·\^=\)":"cat",
        "=_\^=	":"cat",
        "\(\.\.\)":"Looking down",
        "\(\._\.\)":"Looking down",
        "\^m\^":"Giggling with hand covering mouth",
        "\(\・\・?":"Confusion",
        "\(?_?\)":"Confusion",
        ">\^_\^<":"Normal Laugh",
        "<\^!\^>":"Normal Laugh",
        "\^/\^":"Normal Laugh",
        "\（\*\^_\^\*）" :"Normal Laugh",
        "\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
        "\(^\^\)":"Normal Laugh",
        "\(\^\.\^\)":"Normal Laugh",
        "\(\^_\^\.\)":"Normal Laugh",
        "\(\^_\^\)":"Normal Laugh",
        "\(\^\^\)":"Normal Laugh",
        "\(\^J\^\)":"Normal Laugh",
        "\(\*\^\.\^\*\)":"Normal Laugh",
        "\(\^—\^\）":"Normal Laugh",
        "\(#\^\.\^#\)":"Normal Laugh",
        "\（\^—\^\）":"Waving",
        "\(;_;\)/~~~":"Waving",
        "\(\^\.\^\)/~~~":"Waving",
        "\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
        "\(T_T\)/~~~":"Waving",
        "\(ToT\)/~~~":"Waving",
        "\(\*\^0\^\*\)":"Excited",
        "\(\*_\*\)":"Amazed",
        "\(\*_\*;":"Amazed",
        "\(\+_\+\) \(@_@\)":"Amazed",
        "\(\*\^\^\)v":"Laughing,Cheerful",
        "\(\^_\^\)v":"Laughing,Cheerful",
        "\(\(d[-_-]b\)\)":"Headphones,Listening to music",
        '\(-"-\)':"Worried",
        "\(ーー;\)":"Worried",
        "\(\^0_0\^\)":"Eyeglasses",
        "\(\＾ｖ\＾\)":"Happy",
        "\(\＾ｕ\＾\)":"Happy",
        "\(\^\)o\(\^\)":"Happy",
        "\(\^O\^\)":"Happy",
        "\(\^o\^\)":"Happy",
        "\)\^o\^\(":"Happy",
        ":O o_O":"Surprised",
        "o_0":"Surprised",
        "o\.O":"Surpised",
        "\(o\.o\)":"Surprised",
        "oO":"Surprised",
        "\(\*￣m￣\)":"Dissatisfied",
        "\(‘A`\)":"Snubbed or Deflated"}

        for emoticon, description in emot_dict.items():
            text = text.replace(emoticon, description)
        return text

    def number_to_words(self, match):
        """Converts numbers to words."""
        return num2words(int(match.group()))
    def process_nums(self, text):
        # Example regex to find numbers in the text
        return re.sub(r'\d+', self.number_to_words, text)

    def remove_stopwords(self, text):
        """Removes stopwords from text."""
        word_tokens = word_tokenize(text)
        return ' '.join([word for word in word_tokens if word not in stopwords_list])

    # Function to get POS tag for lemmatization
    def get_wordnet_pos(self,word):
        """Map POS tag to first character for lemmatizer."""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV }
        return tag_dict.get(tag, wordnet.NOUN)

    # Function to lemmatize text
    def lemmatize_text(self,text):
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        lemmatized_tokens = [lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in tokens if word.isalpha()]
        return ' '.join(lemmatized_tokens)

    def text_preprocessing(self, text):
        """Applying all preprocessing steps to text."""
        # text = self.adjust_grammar(text)
        # text = self.correct_grammar_with_gpt(text)
        text = re.sub(self.patterns["date"], '', text)
        text = re.sub(self.patterns["time"], '', text)
        text = self.convert_emojis(text)
        text = self.convert_emoticons(text)
        text = re.sub(self.patterns["hyperlink"], '', text)
        text = re.sub(self.patterns["user_mention"], '', text)
        text = re.sub(self.patterns["new_line"], ' ', text)
        text = re.sub(self.patterns["amper_begin_pattern"], '', text)
        text = contractions.fix(text)
        text = self.lemmatize_text(text)
        text = re.sub(self.patterns["multi_space"], ' ', text)
        text = re.sub(self.patterns["special_characters"], ' ', text)
        text = self.process_nums(text)
        text = re.sub(self.patterns["accented"], '', text)
        text = self.remove_stopwords(text)
        text = re.sub(self.patterns["punctuations"], '', text)
        text = text.lower()
        return text.strip()
        
class Analysis:
    def __init__(self, n_grams = [(2, 2), (3, 3), (4, 4)]):
        self.n_grams = n_grams

    def exploratory_data_analysis(self, data, text):
        data['length'] = data[text].str.len()
        data['word_count'] = data[text].apply(lambda x: len(x.split()))
        data['mean_word_length'] = data[text].apply(lambda x: np.mean([len(word) for word in x.split()]))
        data['mean_sent_length'] = data[text].apply(lambda x: np.mean([len(sent) for sent in sent_tokenize(x)]))
        # data.dropna(inplace=True)
        return data

    def frequent_ngrams(self, data, text, top_n = 20):
        for n_gram in self.n_grams:
            cv = CountVectorizer(ngram_range=n_gram)
            ngram_matrix = cv.fit_transform(data[text])
            count_values = ngram_matrix.toarray().sum(axis=0)

            ngram_freq = pd.DataFrame(sorted(
                [(count_values[i], k) for k, i in cv.vocabulary_.items()],
                reverse=True), columns=["frequency", "ngram"])

            sns.barplot(x=ngram_freq['frequency'][:top_n], y=ngram_freq['ngram'][:top_n])
            plt.title(f'Top 10 Most Frequently Occurring {n_gram[0]}-grams')
            plt.xlabel("Frequency")
            plt.ylabel("N-gram")
            plt.show()

    def plot_mean_word_length(self, dataset1, dataset2, name1, name2):
        # Calculate the mean word length for each dataset
        mean_word_length_1 = dataset1['length'].mean()
        mean_word_length_2 = dataset2['length'].mean()
        # Create a DataFrame to hold the results for easy plotting
        results_df = pd.DataFrame({
            "Dataset": [name1, name2],
            "Mean Word Length": [mean_word_length_1, mean_word_length_2]
        })
        # Plot the data
        plt.figure(figsize=(5, 3))
        plt.bar(results_df["Dataset"], results_df["Mean Word Length"], color=['skyblue', 'salmon'])
        plt.xlabel("Dataset")
        plt.ylabel("Mean Word Length")
        plt.title("Mean Word Length Comparison")
        plt.show()


# import os
# import pandas as pd
# import kaggle
# import kagglehub
from objective_1 import Preprocessor, Read, Analysis, BERTanalyzer
from dotenv import load_dotenv
load_dotenv()

def main():
   # Initialize the classes
    preprocessor = Preprocessor()
    reader = Read()
    analyser = Analysis()
    bert_analyzer = BERTanalyzer()

    # Authenticate with Kaggle API
    kaggle.api.authenticate()

    # Read the CSV files into DataFrames
    hate_speech = reader.read_and_filter_dataset('hate_speech/en_hf_102024.csv', 'text')
    fake_news = reader.read_and_filter_dataset('fake_news/WELFake_Dataset.csv', 'text')
    abbrev_profanity = pd.read_excel('abbreviation-list-english-1/abbreviations_eng.xls')

    # Drop missing values
    fake_news.dropna(inplace=True)
    hate_speech.dropna(inplace=True)
    abbrev_profanity.dropna(inplace=True)

    # Extract DataFrame to include only rows with label = 1
    hate_speech_super_df = hate_speech.loc[hate_speech['labels'] == 1, ['text', 'labels']].sample(n=20, random_state=42).reset_index(drop=True)
    fake_news_df = fake_news.loc[fake_news['label'] == 1, ['text', 'label']].sample(n=20, random_state=42).reset_index(drop=True)
    abbrev_profanity = abbrev_profanity[['abbr', 'long']]

    print('\n')
    print('Dataset reader completed!')
    print('\n')

    print('Starting senitment analysis!')
    print('\n')

    # Apply sentiment analysis to a specific column
    hate_speech_super_df['bert_sentiment'] = hate_speech_super_df['text'].apply(bert_analyzer.predict)
    fake_news_df['bert_sentiment'] = fake_news_df['text'].apply(bert_analyzer.predict)
    
    print('*******************************************************')
    print('Senttment analysis completed!')
    print('\n')

    print('*******************************************************')
    print('Plotting sentiment analysis!')
    print('\n')
    #plot the sentiments of the original data
    bert_analyzer.plot_sentiment_distribution(hate_speech_super_df['bert_sentiment'], fake_news_df['bert_sentiment'], 'Hate Speech Super', 'Fake News')

    print('*******************************************************')
    print('Sentiment analysis PLOTTING completed!')
    print('\n')

    print('Cosine similarity computation begins!')
    print('\n')
    #calculate the cosine similarity between our 2 datasets
    average_similarity = bert_analyzer.cosine_similarity(hate_speech_super_df, fake_news_df)
    print('The cosine similairty between hate speech and fake news content is: ', average_similarity)

    
    print('\n')
    print('*******************************************************')
    print('Cosine similarity computation completes!')
    
    print('\n')
    print('*******************************************************')
    print('text preproecessing begins!')
    #preprocess the original datasets
    hate_speech_super_df['cleaned_processed_text'] = hate_speech_super_df['text'].apply(preprocessor.text_preprocessing)
    fake_news_df['cleaned_processed_text'] = fake_news_df['text'].apply(preprocessor.text_preprocessing)

    print('\n')
    print('*******************************************************')
    print('text preproecessing end. here is the hea dof the fake news datasets!')
    print('\n')
    print(fake_news.head())

    #do some prelimiary exploratory dat analysis
    exploratory_hate= analyser.exploratory_data_analysis(hate_speech_super_df, 'cleaned_processed_text')
    exploratory_fake = analyser.exploratory_data_analysis(fake_news_df, 'cleaned_processed_text')

    print('\n')
    print('*******************************************************')
    print('exploratory dataset end. here is the hea dof the fake news datasets!')
    print('\n')
    print(fake_news.head())

    #plot the statistics for wach dataset
    analyser.plot_mean_word_length(hate_speech_super_df, fake_news_df, 'Hate Speech Super', 'Fake News')

    print('\n')
    print('*******************************************************')
    print('plotting mean word length end. here is the hea dof the fake news datasets!')
    print('\n')
    print(fake_news.head())

    #analyse the n_grams for each datase
    analyser.frequent_ngrams(hate_speech_super_df, 'cleaned_processed_text')
    analyser.frequent_ngrams(fake_news_df, 'cleaned_processed_text')
    print('*******************************************************')
    print('\n')
    print("This Script Has finished running!")

if __name__ == "__main__":
    main() 
