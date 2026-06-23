import argparse
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ['Game reaction', 'Analysis / strategy', 'News / injury update', 'Meme / opinion']


def load_model(model_dir: str, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def predict(tokenizer, model, device, texts):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = F.softmax(logits, dim=-1).cpu()
        preds = probs.argmax(dim=-1).cpu().numpy()
    results = []
    for i, text in enumerate(texts):
        p = preds[i]
        prob_list = probs[i].tolist()
        results.append({'text': text, 'pred': int(p), 'label': LABELS[int(p)], 'probs': prob_list})
    return results


def repl(tokenizer, model, device):
    print('Interactive demo. Type a post and press Enter. Empty line to quit.')
    try:
        while True:
            text = input('> ').strip()
            if not text:
                break
            res = predict(tokenizer, model, device, [text])[0]
            print(f"Prediction: {res['label']} (id={res['pred']})")
            for i, lab in enumerate(LABELS):
                print(f"  {i}: {lab} — {res['probs'][i]:.3f}")
    except (EOFError, KeyboardInterrupt):
        print('\nExiting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive demo for TakeMeter model')
    parser.add_argument('--model-dir', default='model_output_real_only')
    parser.add_argument('texts', nargs='*', help='Texts to classify (optional). If omitted, starts interactive REPL')
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_dir)

    if args.texts:
        results = predict(tokenizer, model, device, args.texts)
        for r in results:
            print(f"{r['label']} : {r['text']}")
    else:
        repl(tokenizer, model, device)
