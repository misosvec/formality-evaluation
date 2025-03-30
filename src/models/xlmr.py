from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

# load tokenizer and model weights
tokenizer = XLMRobertaTokenizerFast.from_pretrained('s-nlp/xlmr_formality_classifier')
model = XLMRobertaForSequenceClassification.from_pretrained('s-nlp/xlmr_formality_classifier')

id2formality = {0: "formal", 1: "informal"}
texts = [
    "I like you. I love you",
    "Hey, what's up?",
    "Siema, co porabiasz?",
    "I feel deep regret and sadness about the situation in international politics.",
]

# prepare the input
encoding = tokenizer(
    texts,
    add_special_tokens=True,
    return_token_type_ids=True,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

# inference
output = model(**encoding)

formality_scores = [
    {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
    for text_scores in output.logits.softmax(dim=1)
]
formality_scores
