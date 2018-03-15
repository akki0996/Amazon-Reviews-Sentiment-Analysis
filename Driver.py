import sys
import warnings

from sklearn.feature_extraction.text import CountVectorizer
import pickle
import DataCleaning as DC
import sklearn

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC


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
	else:
		cleaned_reviews = DC.create_cleaned_reviews(DC.reviews())
		save_reviews_to_pickle(cleaned_reviews)

	reviews_text = cleaned_reviews[1]
	reviews_result = cleaned_reviews[2]

	cv = CountVectorizer()
	x = cv.fit_transform(reviews_text[:10000]).toarray()
	y = reviews_result[:10000]

	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=101)

	MNBclassifier = MultinomialNB()
	MNBclassifier.fit(x_train, y_train)

	y_pred = MNBclassifier.predict(x_test)
	# print(confusion_matrix(y_test, y_pred))

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
	print("Naive Bayes - MultinomialNB: ", accuracy)

	# BernoulliNB

	BNBclassifier = BernoulliNB()
	BNBclassifier.fit(x_train, y_train)

	y_pred = BNBclassifier.predict(x_test)

	# print(confusion_matrix(y_test, y_pred))

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
	print("Naive Bayes - BernoulliNB: ", accuracy)

	# Logistic Regression

	LRclassifier = LogisticRegression()
	LRclassifier.fit(x_train, y_train)

	y_pred = LRclassifier.predict(x_test)

	# print(confusion_matrix(y_test, y_pred))

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
	print("Linear Model - Logistic Regression: ", accuracy)

	SGDclassifier = SGDClassifier(max_iter=5, tol=None)
	SGDclassifier.fit(x_train, y_train)

	y_pred = SGDclassifier.predict(x_test)

	# print(confusion_matrix(y_test, y_pred))

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
	print("Linear Model - SGD Classifier: ", accuracy)

	SVCclassifier = SGDClassifier()
	SVCclassifier.fit(x_train, y_train)

	y_pred = SVCclassifier.predict(x_test)

	# print(confusion_matrix(y_test, y_pred))

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
	print("SVM - Linear SVC: ", accuracy, "\n")

	print("Testing classifiers....")
	print("Predicting Sentiment....", "\n")


	InputMNBReviewNegative = "This product was bad and I hated it. It was a worse thing I've ever bought. It is terrible"
	InputMNBReviewPositive = "This product was good and I loved it. It was the best thing I've ever bought."

	print("Positive Review: ", InputMNBReviewPositive)
	print("Negative Review: ", InputMNBReviewNegative, "\n")

	InputMNBReviewPositive = DC.clean_review(InputMNBReviewPositive)
	InputMNBReviewNegative = DC.clean_review(InputMNBReviewNegative)

	testPositive = cv.transform([InputMNBReviewPositive])
	testNegative = cv.transform([InputMNBReviewNegative])

	Positive_Pred = MNBclassifier.predict(testPositive)[0]
	Negative_Pred = MNBclassifier.predict(testNegative)[0]

	print("Naive Bayes MultinomialNB - Testing Positive Review: ", ("Positive" if (Positive_Pred == 1) else "Negative"))
	print("Naive Bayes MultinomialNB - Testing Negative Review: ", ("Positive" if (Negative_Pred == 1) else "Negative"), "\n")

	Positive_Pred = BNBclassifier.predict(testPositive)[0]
	Negative_Pred = BNBclassifier.predict(testNegative)[0]

	print("Naive Bayes BernoulliNB - Testing Positive Review: ", ("Positive" if (Positive_Pred == 1) else "Negative"))
	print("Naive Bayes BernoulliNB - Testing Negative Review: ", ("Positive" if (Negative_Pred == 1) else "Negative"), "\n")

	Positive_Pred = LRclassifier.predict(testPositive)[0]
	Negative_Pred = LRclassifier.predict(testNegative)[0]

	print("Linear Model Logistic Regression - Testing Positive Review: ", ("Positive" if (Positive_Pred == 1) else "Negative"))
	print("Linear Model Logistic Regression - Testing Negative Review: ", ("Positive" if (Negative_Pred == 1) else "Negative"), "\n")

	Positive_Pred = SGDclassifier.predict(testPositive)[0]
	Negative_Pred = SGDclassifier.predict(testNegative)[0]

	print("Linear Model SGDClassifier - Testing Positive Review: ", ("Positive" if (Positive_Pred == 1) else "Negative"))
	print("Linear Model SGDClassifier - Testing Negative Review: ", ("Positive" if (Negative_Pred == 1) else "Negative"), "\n")

	Positive_Pred = SVCclassifier.predict(testPositive)[0]
	Negative_Pred = SVCclassifier.predict(testNegative)[0]

	print("SVM LinearSVC Testing Positive Review: ", ("Positive" if (Positive_Pred == 1) else "Negative"))
	print("SVM LinearSVC Testing Negative Review: ", ("Positive" if (Negative_Pred == 1) else "Negative"), "\n")


if __name__ == '__main__':
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	main()
