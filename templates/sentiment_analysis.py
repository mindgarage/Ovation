from datasets import AmazonReviews
from datasets import HotelReviews

ar = HotelReviews()
#ar.create_vocabulary(min_frequency=10, name='mera_vocab')
ar.train.open()
batch = ar.train.next_batch(raw=True, pad=0, sentence_pad=0, seq_begin=False, one_hot=False)
print('eeithi')