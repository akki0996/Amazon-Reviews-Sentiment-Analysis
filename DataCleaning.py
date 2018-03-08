import json
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

LOWEST_REVIEW_SCORE = 3.0
UNHELPFUL_LIMIT = 5


def reviews():
	review_list = []
	with open('review_video.json') as f:
		for line in f:
			review_list.append(json.loads(line))
	return review_list


def clean_review(text):
	review = re.sub('[^a-zA-Z]', ' ', text)  # punctuation
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()  # stemming
	#review = [ps.stem(word) for word in review]
	review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
	review = " ".join(review)
	return review


def create_cleaned_reviews(the_review_list):
	reviews_text = []
	reviews_result = []

	cleaned_reviews = []

	for json_obj in the_review_list[:10000]:
		is_helpful = True
		pos_or_neg = 0

		if json_obj.get("helpful")[1] > UNHELPFUL_LIMIT:
			is_helpful = False

		if is_helpful:
			overall = json_obj.get("overall")

			if overall > LOWEST_REVIEW_SCORE:
				pos_or_neg = 1

			cleaned_text = clean_review(json_obj.get("reviewText"))
			reviews_text.append(cleaned_text)
			reviews_result.append(pos_or_neg)
			cleaned_reviews.append((cleaned_text, pos_or_neg))

	return cleaned_reviews, reviews_text, reviews_result

# asin = json_obj.get('asin')
# I choose asin as the dictionary key. asin key represents the product number in the amazon
# dict { 'asin': {'positive_text': [], 'negative_text': []}}
