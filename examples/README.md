# Example Texts for Testing

This directory contains sample texts for testing the AI slop detection system.

## Sample Files

- **simple_human.txt** - Casual, natural human writing
- **human_news.txt** - Professional news article
- **famous_speech.txt** - Historical speech (Martin Luther King Jr.)
- **technical_writing.txt** - Technical documentation
- **creative_writing.txt** - Creative prose
- **ai_slop.txt** - AI-generated corporate text
- **corporate_slop.txt** - Corporate buzzword-heavy text
- **extreme_ai_slop.txt** - Heavily AI-generated content

## Usage

```bash
# Test with a sample
python -m sloplint.cli analyze examples/simple_human.txt --explain

# Compare different samples
python -m sloplint.cli analyze examples/human_news.txt examples/ai_slop.txt --explain
```

## Expected Results

- **Human texts** should score lower (Clean/Watch)
- **AI slop texts** should score higher (Sloppy/High-Slop)
- **Confidence** should be high (1.000) for all samples
