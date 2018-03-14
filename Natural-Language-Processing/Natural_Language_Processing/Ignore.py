
import json
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


clean = "Amazon is totally my school supply go to place. They may not have the colors I want, but they ALWAYS have the price! 3rd calculator I've ordered here. Just like what you buy at the store (and usually less expensive)"

review = re.sub('[^a-zA-Z]', ' ', clean)  # punctuation
review = review.lower()
review = review.split()
ps = PorterStemmer()  # stemming
#review = [ps.stem(word) for word in review]
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = " ".join(review)

print(review)