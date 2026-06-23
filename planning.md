Community

I chose r/nba because it is active and has distinct discussion types: reactions, tactics, news, and opinion. That variety makes classification meaningful.

Labels

1. Game reaction
   - Definition: A fan reaction to a recent game event, usually emotional.
   - Examples:
     - "That last second block by Kawhi was insane — best ending to a playoff game all year!"
     - "I can't believe the Knicks blew a 20-point lead in the fourth, this team is so frustrating."

2. Analysis / strategy
   - Definition: A post offering tactical analysis, lineup discussion, or how a team should play.
   - Examples:
     - "The Heat need to switch more on defense and stop letting the big man iso in the post."
     - "If the Lakers run more pick-and-roll with Davis, they can exploit Boston's weak rim protection."

3. News / injury update
   - Definition: A factual post about roster moves, injuries, trades, or official announcements.
   - Examples:
     - "Sources say Durant is dealing with a hamstring issue and might miss Game 3."
     - "Looks like the trade deadline deal is official: team X is sending a first-round pick for guard Y."

4. Meme / opinion
   - Definition: A humorous or general opinion post without a specific game or news focus.
   - Examples:
     - "If the Spurs draft another 7-footer, their team motto should just be 'Bigger, not better.'"
     - "I'm convinced LeBron is playing chess while everyone else is playing checkers."

Hard edge cases

Posts that mix analysis and reaction are the main ambiguity. Example: "The Warriors should run more pick-and-roll, but it's still hilarious how many shots Klay missed." If the main intent is strategy, label it Analysis / strategy. If it is primarily emotional, label it Game reaction. For truly mixed cases, choose the label that is most useful for the tool (e.g. factual update over opinion).

Data collection plan

- Source: r/nba posts via Reddit API or Pushshift, focusing on recent playoffs.
- Quantity: 200 examples total, roughly 50 per label.
- Annotation: human review each post and assign one label.
- Format: single CSV with `text`, `label`, and `notes`; do not pre-split.
- Public data only.
- If a label is under 40 examples after 200 posts, collect additional data with target keywords or broaden the label slightly.

Evaluation metrics

- Accuracy: overall baseline.
- Precision and recall per label: to catch imbalances and label-specific issues.
- Macro F1: equal weight across labels.
- Confusion matrix: find common confusions like Game reaction vs Analysis / strategy.
- Zero-shot baseline: compare to Groq `llama-3.3-70b-versatile` on the same test set.

Definition of success

- Success: >= 80% accuracy, >= 75% macro F1, and no label recall below 65%.
- Acceptable first version: >= 75% accuracy, >= 70% macro F1, and >= 60% recall per label.
- If these thresholds are not met, the model is not ready and requires more annotation or tighter labels.

AI Tool Plan

- Label stress-testing
  - Give the AI label definitions and edge cases, and ask for 5–10 boundary examples.
  - If examples are unclear, tighten the definitions before annotating.

- Annotation assistance
  - Use an LLM to pre-label an initial batch, then review and correct every label.
  - Prefer a strong model like GPT-4.1 or equivalent.
  - Track `prelabel_method` and `reviewed_by` in the dataset.

- Failure analysis
  - Give misclassified examples to an AI tool and ask for error patterns.
  - Look for recurring confusion pairs, slang problems, and bias toward one label.
  - Verify patterns with the confusion matrix and manual review.
