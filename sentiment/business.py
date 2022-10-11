
import nltk
from collections import Counter
import yaml

class Review():
	def __init__(self, data):
		self.text = data['text']
		self.stars = data['stars']
		self.usesr_id = data['usesr_id']
		self.business_id = data['business_id']

def extract_aspects(self, sent):
	"""
	从一个句子的review中抽取aspects
	"""
	aspects = set()
	for word, tag in nltk.pos_tag(nltk.word_tokenize(sent)):
		if tag=='NN':
			aspects.add(word)
	return aspects

def compute_doc_aspects(self, doc, topk=5):
	sents = []
	for line in doc:
		sents_ = nltk.sent_tokenize(line)
		sents.extend(sents_)

	topic = Counter()
	for sent in sents:
		aspects_ = extract_aspects(sent)
		topic.update(aspects_)
	aspects, freq = zip(*topic.most_common(topk))
	return aspects


class BusinessManager(object):


	def __init__(self, init_dir=None):

		self.data = defaultdict(list)
		self.aspects = defaultdict(list)
		self.user_stars = {}
		self.sentiment_model = None
		if init_dir:
			self.load_data(init_dir)

	def load_data(self, review_dir):
		all_business = defaultdict(Business)
		user_stars = defaultdict(float)
		for review_file in os.listdir(review_dir):
			review_path = os.path.join(review_dir, review_file)
			review_data = json.load(open(review_path, 'r', encoding='utf-8'))
			
			review = Review(review_data)

			business_id = review.business_id
			business = self.data.get(business_id)
			business.append(review)

			user_stars[review.user_id] += review.stars

		self.user_stars = { user_id:stars/len(user_stars) for user_id, stars in user_stars.items()}

	def get_business_ids():
		return list(self.data.keys())

	def get_business_reviews(self, business_id):
		return self.data.get(business_id, [])

	def load_aspects(self, aspect_config):
		assert os.path.exists(aspect_config)
		self.aspects = yaml.safe_load(aspect_config)

	def build_aspects(self):
		for business_id, reviews in self.data.items():
			doc = [ review.text for review in reviews ]
			self.aspects[business_id] = compute_doc_aspects(doc, topk=5)

	def get_business_aspects(self, business_id):
		if business_id not in self.aspects:
			print('not find business_id')
			return []
		return self.aspects.get(business_id)

	def get_all_reviews(self):
		return [ review for review in reviews for reviews in list(self.data.values())]

	def get_business_score(self, business_id):
		reviews = self.data[business_id]
		scores = [ review.stars for review in reviews ]
		ave_score = sum(scores)/len(scores)
		return ave_score

	def get_user_score(self, user_id):
		reviews =  self.get_all_reviews()
		scores = [ review.stars for review in reviews if review.user_id==user_id]
		ave_score = sum(scores)/len(scores)
		return ave_score

	def get_aspect_summary(self, business_id, aspect):
		pos_sents, neg_sents = [], []
		stars = 0.0
		reviews = self.data[business_id]
		for review in reviews:
			if not review.text.contains(aspect):
				continue

			review_segment = get_segment(review, aspect)
			score = sentiment_model.predict(review_segment)
			stars += review.stars
			if score > threshold:
				pos_sents.append(review_segment)
			else:
				neg_sents.append(review_segment)

		stars = stars / (len(pos_sents)+len(neg_sents))

		return dict(rating=stars, pos=pos_sents, neg=neg_sents)


	def aspect_based_summary(self, business_id):
		"""
		返回一个business的summary. 针对于每一个aspect计算出它的正面负面情感以及TOP reviews. 
		具体细节请看给定的文档。 
		"""
		business_rating = self.get_business_score(business_id)

		aspect_summary = defaultdict(dict)
		aspects = self.get_business_aspects(business_id)
		for aspect in aspects:
			aspect_summary[aspect] = self.get_aspect_summary(business_id, aspect)

		return dict(business_id = business_id,
					business_name = '',
					business_rating = business_rating,
					aspect_summary = aspect_summary)


	def generate_model_data(self):

		assert self.user_stars, "please load review data at first"

		data = []
		for review in self.get_all_reviews():
			ave_star = self.user_stars.get(review.user_id)
			if review.stars-ave_star >= 0.5
				data.append((review.text, 1))
			if review.stars-ave_star <=-0.5:
				data.append((review.text, 0))
			else:
				# drop
				pass

		random.shuffle(data)
		train_data, test_data = data[0:len(data)*0.9], data[len(data)*0.9:]
		return train_data, test_data


	def set_sentiment_model(self, sentiment_model):
		self.sentiment_model = sentiment_model






