from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle
import DataCleaning as DC
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC, LinearSVC, NuSVC


CLEANED_REVIEWS_PICKLE_FILENAME = "cleaned_reviews.pickle"


# returns true or false depending on user's answers in console
def yes_no(answer):
	yes = set(['yes', 'y', 'ye', ''])
	no = set(['no', 'n'])

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
	if yes_no("Load cleaned reviews from pickle? (Yes, load from pickle or No,"
			" create new set and overwrite current pickle): "):
		cleaned_reviews = get_reviews_from_pickle()
		print("size of sample = " + str(len(cleaned_reviews)))
	else:
		cleaned_reviews = DC.create_cleaned_reviews(DC.reviews())
		save_reviews_to_pickle(cleaned_reviews)
		print("size of sample = " + str(len(cleaned_reviews)))

	reviews_text = cleaned_reviews[1]
	reviews_result = cleaned_reviews[2]

	cv = CountVectorizer()
	x = cv.fit_transform(reviews_text[:10000]).toarray()
	y = reviews_result[:10000]

	print(len(x[0]))

	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=101)

	classifier = LinearSVC()
	classifier.fit(x_train, y_train)

	y_pred = classifier.predict(x_test)

	print(classification_report(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred))

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
	print(accuracy)


if __name__ == '__main__':
	main()
