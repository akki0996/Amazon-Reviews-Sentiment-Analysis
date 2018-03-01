from sklearn.feature_extraction.text import CountVectorizer
import pickle
import DataCleaning as DC

CLEANED_REVIEWS_PICKLE_FILENAME = "cleaned_reviews.pickle"

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
    save_reviews = open(CLEANED_REVIEWS_PICKLE_FILENAME, "wb")
    pickle.dump(the_review_list, save_reviews)
    save_reviews.close()


def main():
    # create cleaned reviews
    if yes_no("Load cleaned reviews from pickle? (Yes, load from pickle or No, "
              "create new set and overwrite current pickle): "):
        cleaned_reviews = get_reviews_from_pickle()
        print("size of sample = " + str(len(cleaned_reviews)))
    else:
        cleaned_reviews = DC.create_cleaned_reviews(DC.reviews())
        save_reviews_to_pickle(cleaned_reviews)
        print("size of sample = " + str(len(cleaned_reviews)))

    corpus = cleaned_reviews[1]
    
    cv = CountVectorizer(max_features=3500)
    X = cv.fit_transform(corpus).toarray()
    y = cleaned_reviews[2]
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(y_pred)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(cm)


# main function, calls the main function if ParseJSON.py file is ran explicitly, rather than as a module
if __name__ == '__main__':
    main()


