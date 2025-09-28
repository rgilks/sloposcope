# Sloposcope Example Texts

This directory contains example texts that demonstrate different types of content for testing the AI slop detection system.

## Example Files

- **`ai_slop.txt`** - Classic AI-generated content with excessive verbosity, corporate jargon, and repetitive patterns
- **`corporate_slop.txt`** - Business-style content with management speak and buzzwords
- **`creative_writing.txt`** - Human-written creative content that should score as "clean"
- **`extreme_ai_slop.txt`** - Highly problematic AI content with maximum slop characteristics
- **`famous_speech.txt`** - Historical human speech content for comparison
- **`human_news.txt`** - Professional journalism that should score well
- **`simple_human.txt`** - Basic human writing for baseline testing
- **`technical_writing.txt`** - Technical documentation that should be relatively clean

## Usage

Use these files to test the system:

```bash
# Test with obvious AI slop
sloposcope analyze examples/ai_slop.txt --explain

# Test with human content
sloposcope analyze examples/human_news.txt --explain

# Compare results across different content types
sloposcope analyze examples/*.txt --json results.json
```

## Expected Results

- **AI-generated examples**: Should score > 0.70 (Sloppy/High-Slop)
- **Human examples**: Should score < 0.50 (Clean)
- **Mixed content**: Will vary based on specific characteristics
