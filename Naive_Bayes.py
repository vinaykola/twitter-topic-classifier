""" This program uses the Naive Bayes Classifier to classify the validation set 
after training on the training set. """

import string

# Extract actual necessary words from the tweet
def extract_words(tweet_words):
	words = []
	alpha_lower = string.ascii_lowercase
	alpha_upper = string.ascii_uppercase
	numbers = [str(n) for n in range(10)]
	for word in tweet_words:
		cur_word = ''
		for c in word:
			if (c not in alpha_lower) and (c not in alpha_upper) and (c not in numbers):
				if len(cur_word) >= 2:
					words.append(cur_word.lower())
				cur_word = ''
				continue
			cur_word += c
		if len(cur_word) >= 2:
			words.append(cur_word.lower())
	return words

# Get Training Data from the input file
def get_training_data():
	f = open('training.txt', 'r')
	training_data = []
	for l in f.readlines():
		l = l.strip()
		tweet_details = l.split()
		tweet_id = tweet_details[0]
		tweet_label = tweet_details[1]
		tweet_words = extract_words(tweet_details[2:])
		training_data.append([tweet_id, tweet_label, tweet_words])
	
	f.close()
	
	return training_data

# Get Test Data from the input file
def get_test_data():
	f = open('test.txt', 'r')
	validation_data = []
	for l in f.readlines():
		l = l.strip()
		tweet_details = l.split(' ')
		tweet_id = tweet_details[0]
		tweet_words = extract_words(tweet_details[1:])
		validation_data.append([tweet_id, '', tweet_words])

	f.close()

	return validation_data

# Get list of words in the training data
def get_words(training_data):
	words = []
	for data in training_data:
		words.extend(data[2])
	return list(set(words))

# Get Probability of each word in the training data
# If label is specified, find the probability of each word in the corresponding labelled tweets only
def get_word_prob(training_data, label = None):
	words = get_words(training_data)
	freq = {}

	for word in words:
		freq[word] = 1

	total_count = 0
	for data in training_data:
		if data[1] == label or label == None:
			total_count += len(data[2])
			for word in data[2]:
				freq[word] += 1

	prob = {}
	for word in freq.keys():
		prob[word] = freq[word]*1.0/total_count

	return prob

# Get Probability of given label
def get_label_count(training_data, label):
	count = 0
	total_count = 0
	for data in training_data:
		total_count += 1
		if data[1] == label:
			count += 1
	return count*1.0/total_count

# Label the test data given the trained parameters Using Naive Bayes Model
def label_data(test_data, sports_word_prob, politics_word_prob, sports_prob, politics_prob):
	labels = []
	for data in test_data:
		data_prob_sports = sports_prob
		data_prob_politics = politics_prob
		
		for word in data[2]:
			if word in sports_word_prob:
				data_prob_sports *= sports_word_prob[word]
				data_prob_politics *= politics_word_prob[word]
			else:
				continue

		if data_prob_sports >= data_prob_politics:
			labels.append([data[0], 'Sports', data_prob_sports, data_prob_politics])
		else:
			labels.append([data[0], 'Politics', data_prob_sports, data_prob_politics])

	return labels

# Print the labelled test data
def print_labelled_data(labels):
	f_out = open('test_labelled.txt', 'w')
	for [tweet_id, label, prob_sports, prob_politics] in labels:
		f_out.write('%s %s\n' % (tweet_id, label))

	f_out.close()


# Get the training and test data
training_data = get_training_data()
test_data = get_test_data()

# Get the probabilities of each word overall and in the two labels
word_prob = get_word_prob(training_data)
sports_word_prob = get_word_prob(training_data, 'Sports')
politics_word_prob = get_word_prob(training_data, 'Politics')

# Get the probability of each label
sports_prob = get_label_count(training_data, 'Sports')
politics_prob = get_label_count(training_data, 'Politics')

# Normalise for stop words
for (word, prob) in word_prob.iteritems():
	sports_word_prob[word] /= prob
	politics_word_prob[word] /= prob

# Label the test data and print it
test_labels = label_data(test_data, sports_word_prob, politics_word_prob, sports_prob, politics_prob)
print_labelled_data(test_labels)
