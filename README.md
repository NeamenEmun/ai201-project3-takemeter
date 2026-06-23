# TakeMeter — Project 3

## Community choice
- r/nba is a strong fit because the subreddit contains real fan reactions, trade rumors, lineup analysis, and player/injury news. Its conversational format and varied post types make it a good domain for a multiclass classifier that separates emotional reaction, tactical analysis, factual update, and general opinion.

## Labels
1. Game reaction — expresses fan emotion or opinion about a game, result, or team performance. Example: "What a finish! {player} just hit a dagger and the crowd lost it." Example: "Unreal comeback by the {team} — down by 15 and won in OT.".
2. Analysis / strategy — explains how lineups, rotations, matchups, or tactics affect outcomes. Example: "If {team} moves to a small-ball lineup with {player} at center, they could outscore opponents in transition." Example: "Switching to a zone defense against {opponent} would neutralize their pick-and-roll advantage.".
3. News / injury update — reports roster changes, trades, injuries, or official team decisions. Example: "BREAKING: {player} has agreed to a one-year deal with the {team}, sources say." Example: "{team} announce that {player} will miss the season due to a torn ACL.".
4. Meme / opinion — general NBA opinion, memes, or commentary not tied to a single factual event or tactical prescription. Example: "Just wanted to show you the malpractice of whatever the fuck the Sacramento Kings are doing." Example: "Tatum + Giannis might be a more dominant duo.".

See `planning.md` for full definitions, edge cases, and the hardest anticipated boundary between labels.

- File: `nba_dataset_submission.csv` (370 examples after augmentation, columns: `text,label,notes`).
- File: `nba_dataset_submission_real_only.csv` (253 real Reddit examples only; no synthetic examples in `notes`).
- File: `nba_dataset_labeled_cleaned.csv` includes the same cleaned examples plus provenance notes.
- Synthetic examples were added to improve label coverage and balance. They are not hidden in the source scripts; the data pipeline preserves provenance via the `notes` field.

## Annotation process
- Used `auto_label_and_export.py` for initial pre-labeling, then reviewed labels manually.
- Data was normalized and deduped with `fix_nba_csv.py`.
- Synthetic augmentation was added with `add_synthetic_posts.py` to ensure sufficient coverage of `News / injury update` and `Analysis / strategy` labels.

## AI Tool Plan
- Label stress-testing: `label_stresstest.txt` contains 10 boundary cases to refine label definitions.
- Annotation assistance: the assistant generated a synthetic data batch and a rule-based pre-labeling script, then reviewed the final cleaned labels.
- Failure analysis: evaluation artifacts will include error examples and an analysis of systematic misclassifications.

## Training plan
- Base model: `distilbert-base-uncased`.
- Platform: Google Colab (recommended T4 GPU).
- Key hyperparameters: 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01. These settings balance speed and generalization for a small dataset.

## Baseline
- Baseline approach: zero-shot classification with a prompt describing the four labels and asking the model to choose one label per post.
- Evaluation will compare the baseline and fine-tuned results on the same test set.

## Final evaluation
- Real-only dataset counts: `Meme / opinion` 223, `News / injury update` 26, `Game reaction` 3, `Analysis / strategy` 1.
- Full submission dataset counts: `Meme / opinion` 240, `News / injury update` 66, `Analysis / strategy` 41, `Game reaction` 23.
- Final fine-tuned model evaluation on `nba_dataset_submission_real_only.csv` (real-only holdout split):
	- Test accuracy: `0.8077`
	- Test macro F1: `0.2234`
	- Confusion matrix (test set): `[[0, 0, 1], [0, 0, 4], [0, 0, 21]]`
	- Note: the real-only test split is small and heavily imbalanced, which makes per-class performance unstable.

## Next steps
1. Review the real-only dataset and label distribution for edge cases.
2. Refine synthetic augmentation and label balance if stronger generalization is needed.
3. Add error analysis on misclassified examples from `model_output_real_only` predictions.
4. Create a demo script or recording showing model predictions and one incorrect prediction explained.
