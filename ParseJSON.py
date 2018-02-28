import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle


UNHELPFUL_LIMIT = 5
LOWEST_REVIEW_SCORE = 3
CLEANED_REVIEWS_PICKLE_FILENAME = "cleaned_reviews.pickle"

review_list = []
positive_review_text_list = []
negative_review_text_list = []

review_list_dict = dict()

# Clean reviews
def clean_review(text):
    review = re.sub('[^a-zA-Z]', ' ', text)  # punctuation
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]  # stop words
    ps = PorterStemmer()  # stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    return review

# returns true or false depending on user's answers in console
def yes_no(answer):
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
     
    while True:
        choice = input(answer).lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        else:
           print("Please respond with 'yes' or 'no'\n")

# gets and returns cleaned reviews list from local .pickle file
def get_reviews_from_pickle():
    # retrieve previous trained classifer results from stored file
    reviews_f = open(CLEANED_REVIEWS_PICKLE_FILENAME, "rb")
    returned_reviews = pickle.load(reviews_f)
    reviews_f.close()

    return returned_reviews


# saves cleaned reviews lists to a local pickle file
def save_reviews_to_pickle(the_review_list):
    # dump reviews results in local pickle file
    save_reviews = open(CLEANED_REVIEWS_PICKLE_FILENAME,"wb")
    pickle.dump(the_review_list, save_reviews)
    save_reviews.close()


# creates a list of cleaned reviews, given list of reviews with cleaned text and pos/neg tuple
def create_cleaned_reviews(the_review_list):
    cleaned_reviews = []
    for json_obj in the_review_list[:500]:
        isHelpful = True

        cleaned_text = clean_review(json_obj.get("reviewText"))
        overall = json_obj.get("overall")
        pos_or_neg = 0
        # if review overall is greater than 3 review = POSITIVE
        if overall > LOWEST_REVIEW_SCORE:
            pos_or_neg = 1

        # if review wasn't helpful to more than 5 people = THROW AWAy
        if json_obj.get("helpful")[1] > UNHELPFUL_LIMIT:
            isHelpful = False

        # check if review should be thrown away
        if isHelpful:
            rated_review_t = (cleaned_text, pos_or_neg)
            cleaned_reviews.append(rated_review_t)

    print("size of sample = " + str(len(cleaned_reviews)))

    return cleaned_reviews
    # asin = json_obj.get('asin')

def main():
    with open('review_video.json') as f:
        for line in f:
            review_list.append(json.loads(line))

    for json_obj in review_list:
        asin = json_obj.get('asin')
        if asin not in review_list_dict:
            review_list_dict[asin] = dict()

        if 'negative_text' not in review_list_dict.get(asin):
            review_list_dict.get(asin)['negative_text'] = []

        if 'positive_text' not in review_list_dict.get(asin):
            review_list_dict.get(asin)['positive_text'] = []

        if json_obj.get('overall') > LOWEST_REVIEW_SCORE:
            review_list_dict.get(asin).get('positive_text').append(json_obj.get('reviewText'))
        else:
            review_list_dict.get(asin).get('negative_text').append(json_obj.get('reviewText'))

        # I choose asin as the dictionary key. asin key represents the product number in the amazon
        # dict { 'asin': {'positive_text': [], 'negative_text': []}}

    # create cleaned reviews
    if yes_no("Load cleaned reviews from pickle? (Yes, load from pickle or No, create new set and overwrite current pickle): "):
        cleaned_reviews = get_reviews_from_pickle()
        print("size of sample = " + str(len(cleaned_reviews)))
    else:
        cleaned_reviews = create_cleaned_reviews(review_list)
        save_reviews_to_pickle(cleaned_reviews)
        print("size of sample = " + str(len(cleaned_reviews)))
   
# main function, calls the main function if ParseJSON.py file is ran explicitly, rather than as a module
if __name__ == '__main__':
    main()
