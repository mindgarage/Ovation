# We put here the default parameters for all datasets. For these parameters, we
# precreated the datasets and organized them already in a directory structure.
#

shuffle = True
stratified = True
tokenizer = 'spacy_tokenizer'
train_validate_split = 0.9
test_split_large = 0.3
test_split_small = 0.2

from microsoft_paraphrase_dataset import MicrosoftParaphraseDataset
from gersen import Gersen

