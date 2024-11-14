# !conda activate mikel_2024

import topmost
from fastopic import FASTopic
import pandas as pd

class Read:
    @staticmethod
    def read_and_filter_dataset (filepath, text_column, max_length=1000):
        data = pd.read_csv(filepath, low_memory=False)
        # Filter rows based on text length
        data = data[data[text_column].str.len() <= max_length]
        return data


class Topic:
    def __init__(self, num_topics=50):
        self.num_topics = num_topics

    def topic_modelling(self, dataset, text):
        # Select the first 100 rows of the 'text' column
        docs = dataset[text].tolist() 

        # Initialize and fit the topic model
        topic_model = FASTopic(num_topics=self.num_topics, verbose=True)
        topic_top_words, doc_topic_dist = topic_model.fit_transform(docs)

        # Visualize topic hierarchy and topic weights
        fig_hierarchy = topic_model.visualize_topic_hierarchy()
        fig_hierarchy.show()
        
        fig_weights = topic_model.visualize_topic_weights(top_n=20, height=500)
        fig_weights.show()

        return topic_top_words, doc_topic_dist, fig_hierarchy, fig_weights

    # def preprocess_text_column(self, dataset, text_column='text', vocab_size=10000, stopwords='English'):
    #     """
    #     Extracts and preprocesses the text column from the dataset.
    #     """
    #     # Select only the first 100 rows of the specified text column
    #     docs = dataset[text_column].head(100).dropna().tolist()

    #     # Preprocessing setup
    #     preprocessing = Preprocessing(vocab_size=vocab_size, stopwords=stopwords)
    #     processed_docs = preprocessing.fit_transform(docs)  # Assuming fit_transform handles processing

    #     return processed_docs

    # def topic_modelling(self, dataset_path, text_column='text', vocab_size=10000, stopwords='English'):
    #     """
    #     Runs topic modeling on the specified text column of the first 100 rows of the dataset.
    #     """
    #     dataset = topmost.data.DynamicDataset(dataset_path, as_tensor=False)

    #     # Preprocess the dataset to get cleaned text data from the specified text column
    #     docs = self.preprocess_text_column(dataset, text_column, vocab_size, stopwords)

    #     # Fit the topic model
    #     topic_model = FASTopic(num_topics=self.num_topics, verbose=True)
    #     topic_top_words, doc_topic_dist = topic_model.fit_transform(docs)

    #     # Visualize topic hierarchy and topic weights
    #     hierarchy_fig = topic_model.visualize_topic_hierarchy()
    #     hierarchy_fig.show()

    #     weights_fig = topic_model.visualize_topic_weights(top_n=20, height=500)
    #     weights_fig.show()

    #     return topic_top_words, doc_topic_dist, hierarchy_fig, weights_fig
def main():
    topic_model = Topic()
    reader = Read()

    fake_news = reader.read_and_filter_dataset('fake_news/WELFake_Dataset.csv', 'text')
    hate_speech = reader.read_and_filter_dataset('hate_speech/en_hf_102024.csv', 'text')

    print('Dataset reader completed!')
    print('\n')

    fake_news.dropna(inplace=True)
    hate_speech.dropna(inplace=True)

    fake_news_df = fake_news.loc[fake_news['label'] == 1, ['text', 'label']].sample(n=200, random_state=42).reset_index(drop=True)
    hate_speech_super_df = hate_speech.loc[hate_speech['labels'] == 1, ['text', 'labels']].sample(n=200, random_state=42).reset_index(drop=True)
    
    print('Dataset collection completed!')

    print('\n')

    print('The code for topic modelling begins here...!')
    print('\n')

    top_topics_fake, topic_distribution_fake, figure_heirarchy_fake, figure_weights_fake = topic_model.topic_modelling (fake_news_df, 'text')
    top_topics_hate, topic_distribution_hate, figure_heirarchy_hate, figure_weights_hate = topic_model.topic_modelling (fake_news_df, 'text')

    print('\n')
    print("Script Has finished running!")
if __name__ == "__main__":
    main() 

