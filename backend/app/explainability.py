import re

import torch
from captum.attr import Saliency


def explain_text(article, tokenizer, model, top_k=8):
    """
    Generate token-level explanations using Captum Saliency.

    Parameters:
        article: Input news article as a string
        tokenizer: DistilBERT tokenizer
        model: Fine-tuned DistilBERT model
        top_k: Number of important tokens to return

    Returns:
        Prediction class, confidence, and important tokens
    """

    model.eval()

    # =========================================================
    # 1. Tokenize the article
    # =========================================================

    encoded = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # =========================================================
    # 2. Get model prediction
    # =========================================================

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

    # =========================================================
    # 3. Convert input token IDs into embeddings
    # =========================================================

    embeddings = model.distilbert.embeddings.word_embeddings(
        input_ids
    )

    embeddings = embeddings.detach().clone()

    embeddings.requires_grad_(True)

    # =========================================================
    # 4. Define forward function for Captum
    # =========================================================

    def forward_func(inputs_embeds):

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        return outputs.logits

    # =========================================================
    # 5. Apply Captum Saliency
    # =========================================================

    saliency = Saliency(forward_func)

    attributions = saliency.attribute(
        embeddings,
        target=predicted_class
    )

    # =========================================================
    # 6. Calculate importance for every token
    # =========================================================

    token_scores = attributions.abs().sum(
        dim=-1
    ).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )

    # =========================================================
    # 7. Remove unwanted tokens
    # =========================================================

    stopwords = {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "of",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "for",
        "with",
        "by",
        "from",
        "as",
        "this",
        "that",
        "it",
        "be",
        "has",
        "have",
        "had"
    }

    explanations = []

    for token, score in zip(tokens, token_scores):

        # Ignore special tokens such as [CLS] and [SEP]
        if token in tokenizer.all_special_tokens:
            continue

        # Ignore punctuation
        if re.fullmatch(r"[^\w]+", token):
            continue

        # Remove DistilBERT subword marker
        clean_token = token.replace("##", "")

        # Ignore common stopwords
        if clean_token.lower() in stopwords:
            continue

        importance = float(score.item())

        explanations.append({
            "token": clean_token,
            "importance": round(importance, 6),
            "effect": "important for prediction"
        })

    # =========================================================
    # 8. Sort tokens by importance
    # =========================================================

    explanations = sorted(
        explanations,
        key=lambda item: item["importance"],
        reverse=True
    )

    # =========================================================
    # 9. Normalize importance scores between 0 and 1
    # =========================================================

    if explanations:

        max_score = max(
            item["importance"]
            for item in explanations
        )

        if max_score > 0:

            for item in explanations:

                item["importance"] = round(
                    item["importance"] / max_score,
                    4
                )

    # =========================================================
    # 10. Return explanation
    # =========================================================

    return {
        "predicted_class": predicted_class,

        "confidence": round(
            float(
                probabilities[predicted_class].item()
                * 100
            ),
            2
        ),

        "top_features": explanations[:top_k]
    }